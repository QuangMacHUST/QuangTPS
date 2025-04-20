#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan Review Panel for QuangTPS.

This module implements an Eclipse-like plan review and approval interface for QuangTPS,
allowing users to review plans, their approval status, and manage the approval workflow.
"""

import os
import logging
import datetime
from typing import List, Dict, Optional, Any

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QSplitter,
    QGroupBox,
    QFormLayout,
    QTextEdit,
    QTabWidget,
    QComboBox,
    QMessageBox,
    QHeaderView,
    QFrame,
    QMenu,
    QAction,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QColor, QBrush

from quangtps.planning.plan_approval import (
    ApprovalRole,
    ApprovalStatus,
    ApprovalAction,
    PlanApprovalWorkflow,
    get_plan_approval_manager,
)
from quangtps.common.user import User, UserRole, get_current_user
from quangtps.planning.treatment_plan import TreatmentPlan
from quangtps.ui.dialogs.plan_approval_dialog import PlanApprovalDialog
from quangtps.ui.mpr_viewer import MPRViewer
from quangtps.ui.dvh_view import DVHView

logger = logging.getLogger(__name__)


class PlanStatusWidget(QWidget):
    """
    Widget to display the current approval status of a plan with color indicator.

    Attributes
    ----------
    _status : ApprovalStatus
        Current approval status
    status_indicator : QLabel
        Visual indicator showing status color
    status_label : QLabel
        Text label showing status name
    """

    # Color mapping for different approval statuses
    STATUS_COLORS = {
        ApprovalStatus.DRAFT: QColor(240, 240, 240),  # Light Gray
        ApprovalStatus.PLANNING: QColor(255, 200, 0),  # Yellow
        ApprovalStatus.PENDING_APPROVAL: QColor(0, 120, 215),  # Blue
        ApprovalStatus.APPROVED_BY_PLANNER: QColor(144, 238, 144),  # Light Green
        ApprovalStatus.APPROVED_BY_PHYSICIST: QColor(60, 179, 113),  # Medium Green
        ApprovalStatus.APPROVED_BY_PHYSICIAN: QColor(0, 158, 115),  # Dark Green
        ApprovalStatus.TREATMENT_APPROVED: QColor(0, 100, 0),  # Very Dark Green
        ApprovalStatus.REJECTED: QColor(213, 94, 0),  # Red/Orange
        ApprovalStatus.UNDER_REVISION: QColor(255, 165, 0),  # Orange
        ApprovalStatus.ARCHIVED: QColor(128, 128, 128),  # Gray
    }

    def __init__(self, status=ApprovalStatus.DRAFT, parent=None):
        """
        Initialize the status widget.

        Parameters
        ----------
        status : ApprovalStatus, optional
            Initial status, by default ApprovalStatus.DRAFT
        parent : QWidget, optional
            Parent widget, by default None
        """
        super().__init__(parent)
        self._status = status

        layout = QHBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(5)

        self.status_indicator = QLabel()
        self.status_indicator.setFixedSize(16, 16)
        self.status_indicator.setStyleSheet(
            f"background-color: {self.STATUS_COLORS[self._status].name()}; "
            f"border-radius: 8px; border: 1px solid #888888;"
        )

        self.status_label = QLabel(self._status.value)

        layout.addWidget(self.status_indicator)
        layout.addWidget(self.status_label)
        layout.addStretch()

        self.setLayout(layout)

    def set_status(self, status: ApprovalStatus) -> None:
        """
        Update the status indicator and label.

        Parameters
        ----------
        status : ApprovalStatus
            New status value
        """
        self._status = status
        self.status_indicator.setStyleSheet(
            f"background-color: {self.STATUS_COLORS[status].name()}; "
            f"border-radius: 8px; border: 1px solid #888888;"
        )
        self.status_label.setText(status.value)


class PlanApprovalHistory(QWidget):
    """
    Widget to display approval history for a treatment plan.

    Attributes
    ----------
    workflow : PlanApprovalWorkflow
        The approval workflow for the current plan
    history_table : QTableWidget
        Table displaying approval history events
    """

    def __init__(self, parent=None):
        """
        Initialize the approval history widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget, by default None
        """
        super().__init__(parent)
        self.workflow = None

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Title
        title_label = QLabel("Approval History")
        title_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(title_label)

        # Table for history
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels(
            ["Date/Time", "User", "Role", "Action", "Status Change", "Comment"]
        )
        self.history_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.history_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.Stretch
        )
        self.history_table.setAlternatingRowColors(True)

        layout.addWidget(self.history_table)
        self.setLayout(layout)

    def set_workflow(self, workflow: PlanApprovalWorkflow) -> None:
        """
        Set the approval workflow and update the history display.

        Parameters
        ----------
        workflow : PlanApprovalWorkflow
            Approval workflow to display
        """
        self.workflow = workflow
        self._populate_history()

    def _populate_history(self) -> None:
        """Populate the history table with events from the workflow."""
        self.history_table.setRowCount(0)

        if not self.workflow:
            return

        # Get event history in chronological order (newest first)
        history = self.workflow.get_history()
        history.reverse()

        for i, event_dict in enumerate(history):
            self.history_table.insertRow(i)

            # Date/Time
            timestamp_item = QTableWidgetItem(
                datetime.datetime.fromisoformat(event_dict["timestamp"]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
            timestamp_item.setTextAlignment(Qt.AlignCenter)
            self.history_table.setItem(i, 0, timestamp_item)

            # User
            user_item = QTableWidgetItem(event_dict["user"])
            self.history_table.setItem(i, 1, user_item)

            # Role
            role_item = QTableWidgetItem(event_dict["role"])
            self.history_table.setItem(i, 2, role_item)

            # Action
            action_item = QTableWidgetItem(event_dict["action"])
            self.history_table.setItem(i, 3, action_item)

            # Status Change
            status_change_item = QTableWidgetItem(
                f"{event_dict['old_status']} → {event_dict['new_status']}"
            )
            self.history_table.setItem(i, 4, status_change_item)

            # Comment
            comment_item = QTableWidgetItem(event_dict.get("comment", ""))
            self.history_table.setItem(i, 5, comment_item)


class ReviewActionPanel(QWidget):
    """
    Panel for performing approval actions on a plan.

    Attributes
    ----------
    workflow : PlanApprovalWorkflow
        The approval workflow for the current plan
    action_combo : QComboBox
        Dropdown for selecting action to perform
    comment_edit : QTextEdit
        Text area for entering action comments
    execute_btn : QPushButton
        Button to execute the selected action
    """

    action_executed = pyqtSignal()

    def __init__(self, parent=None):
        """
        Initialize the review action panel.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget, by default None
        """
        super().__init__(parent)
        self.workflow = None
        self.current_user = get_current_user()

        layout = QVBoxLayout()

        # Title
        title_label = QLabel("Plan Review Actions")
        title_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(title_label)

        # Form for action and comment
        form_layout = QFormLayout()

        self.action_combo = QComboBox()
        self.action_combo.currentIndexChanged.connect(self._on_action_changed)

        self.comment_edit = QTextEdit()
        self.comment_edit.setPlaceholderText("Enter comment for this action...")
        self.comment_edit.setMaximumHeight(100)

        form_layout.addRow("Action:", self.action_combo)
        form_layout.addRow("Comment:", self.comment_edit)

        action_widget = QWidget()
        action_widget.setLayout(form_layout)
        layout.addWidget(action_widget)

        # Buttons
        button_layout = QHBoxLayout()

        self.execute_btn = QPushButton("Execute Action")
        self.execute_btn.setIcon(QIcon.fromTheme("dialog-ok"))
        self.execute_btn.clicked.connect(self._execute_action)

        button_layout.addStretch()
        button_layout.addWidget(self.execute_btn)

        layout.addLayout(button_layout)
        layout.addStretch()

        self.setLayout(layout)

    def set_workflow(self, workflow: PlanApprovalWorkflow) -> None:
        """
        Set the approval workflow and update available actions.

        Parameters
        ----------
        workflow : PlanApprovalWorkflow
            Approval workflow to use
        """
        self.workflow = workflow
        self._update_available_actions()

    def _update_available_actions(self) -> None:
        """Update the action dropdown based on current user role and plan status."""
        self.action_combo.clear()

        if not self.workflow or not self.current_user:
            self.execute_btn.setEnabled(False)
            return

        # Get current user role
        user_role = ApprovalRole(self.current_user.role.name)
        current_status = self.workflow.status

        # Determine available actions based on role and current status
        available_actions = []

        # Planner actions
        if user_role == ApprovalRole.PLANNER:
            if current_status in [
                ApprovalStatus.DRAFT,
                ApprovalStatus.PLANNING,
                ApprovalStatus.UNDER_REVISION,
            ]:
                available_actions.append(ApprovalAction.SUBMIT)
            if current_status == ApprovalStatus.PENDING_APPROVAL:
                available_actions.append(ApprovalAction.RETURN_TO_PLANNING)

        # Physicist actions
        if user_role == ApprovalRole.PHYSICIST:
            if current_status == ApprovalStatus.PENDING_APPROVAL:
                available_actions.append(ApprovalAction.APPROVE)
                available_actions.append(ApprovalAction.REJECT)

            # Allow physicists to return plans to planning in appropriate statuses
            if current_status in [
                ApprovalStatus.PENDING_APPROVAL,
                ApprovalStatus.APPROVED_BY_PLANNER,
            ]:
                available_actions.append(ApprovalAction.RETURN_TO_PLANNING)

        # Physician actions
        if user_role == ApprovalRole.PHYSICIAN:
            if current_status in [
                ApprovalStatus.APPROVED_BY_PHYSICIST,
                ApprovalStatus.PENDING_APPROVAL,
            ]:
                available_actions.append(ApprovalAction.APPROVE)
                available_actions.append(ApprovalAction.REJECT)

            # Allow physicians to return plans to planning in appropriate statuses
            if current_status in [
                ApprovalStatus.PENDING_APPROVAL,
                ApprovalStatus.APPROVED_BY_PLANNER,
                ApprovalStatus.APPROVED_BY_PHYSICIST,
            ]:
                available_actions.append(ApprovalAction.RETURN_TO_PLANNING)

        # Administrator actions (can perform all actions)
        if user_role == ApprovalRole.ADMINISTRATOR:
            for action in ApprovalAction:
                available_actions.append(action)

        # Add available actions to combo box
        for action in available_actions:
            self.action_combo.addItem(action.value, action)

        # Enable or disable execute button
        self.execute_btn.setEnabled(len(available_actions) > 0)

    def _on_action_changed(self, index: int) -> None:
        """
        Handle changes to the selected action.

        Parameters
        ----------
        index : int
            New selected index
        """
        if index < 0:
            return

        action = self.action_combo.itemData(index)

        # Update placeholder text based on action
        if action == ApprovalAction.APPROVE:
            self.comment_edit.setPlaceholderText("Enter approval comments...")
        elif action == ApprovalAction.REJECT:
            self.comment_edit.setPlaceholderText("Enter rejection reason...")
        elif action == ApprovalAction.RETURN_TO_PLANNING:
            self.comment_edit.setPlaceholderText(
                "Enter reason for returning to planning..."
            )
        else:
            self.comment_edit.setPlaceholderText("Enter comment for this action...")

    def _execute_action(self) -> None:
        """Execute the selected action on the workflow."""
        if not self.workflow or not self.current_user:
            return

        current_index = self.action_combo.currentIndex()
        if current_index < 0:
            return

        action = self.action_combo.itemData(current_index)
        comment = self.comment_edit.toPlainText()

        # Convert QT user role to approval role
        user_role = ApprovalRole(self.current_user.role.name)

        success = False

        # Execute appropriate action based on selection
        if action == ApprovalAction.SUBMIT:
            success = self.workflow.submit_for_approval(
                self.current_user.username, user_role, comment
            )
        elif action == ApprovalAction.APPROVE:
            success = self.workflow.approve(
                self.current_user.username, user_role, comment
            )
        elif action == ApprovalAction.REJECT:
            success = self.workflow.reject(
                self.current_user.username, user_role, comment
            )
        elif action == ApprovalAction.RETURN_TO_PLANNING:
            success = self.workflow.return_to_planning(
                self.current_user.username, user_role, comment
            )
        elif action == ApprovalAction.ARCHIVE:
            success = self.workflow.archive(
                self.current_user.username, user_role, comment
            )
        elif action == ApprovalAction.RESTORE:
            success = self.workflow.restore(
                self.current_user.username, user_role, comment
            )

        # Show result message
        if success:
            QMessageBox.information(
                self,
                "Action Executed",
                f"Successfully performed {action.value} action.",
            )
            self.comment_edit.clear()
            self._update_available_actions()
            self.action_executed.emit()
        else:
            QMessageBox.warning(
                self,
                "Action Failed",
                f"Failed to perform {action.value} action. Check permissions and plan status.",
            )


class PlanReviewSummary(QWidget):
    """
    Widget to display a summary of the plan for review.

    Attributes
    ----------
    plan : TreatmentPlan
        The treatment plan being reviewed
    """

    def __init__(self, parent=None):
        """
        Initialize the plan review summary widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget, by default None
        """
        super().__init__(parent)
        self.plan = None

        layout = QVBoxLayout()

        # Title
        title_label = QLabel("Plan Details")
        title_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(title_label)

        # Form layout for plan details
        form_layout = QFormLayout()

        self.plan_id_label = QLabel("--")
        self.plan_name_label = QLabel("--")
        self.plan_description_label = QLabel("--")
        self.plan_technique_label = QLabel("--")
        self.plan_intent_label = QLabel("--")
        self.primary_target_label = QLabel("--")
        self.prescription_label = QLabel("--")
        self.fractions_label = QLabel("--")
        self.plan_date_label = QLabel("--")

        form_layout.addRow("Plan ID:", self.plan_id_label)
        form_layout.addRow("Name:", self.plan_name_label)
        form_layout.addRow("Description:", self.plan_description_label)
        form_layout.addRow("Technique:", self.plan_technique_label)
        form_layout.addRow("Intent:", self.plan_intent_label)
        form_layout.addRow("Primary Target:", self.primary_target_label)
        form_layout.addRow("Prescription:", self.prescription_label)
        form_layout.addRow("Fractions:", self.fractions_label)
        form_layout.addRow("Creation Date:", self.plan_date_label)

        details_widget = QWidget()
        details_widget.setLayout(form_layout)
        layout.addWidget(details_widget)

        layout.addStretch()
        self.setLayout(layout)

    def set_plan(self, plan: TreatmentPlan) -> None:
        """
        Set the plan and update the display.

        Parameters
        ----------
        plan : TreatmentPlan
            Plan to display
        """
        self.plan = plan
        self._update_display()

    def _update_display(self) -> None:
        """Update the display with plan details."""
        if not self.plan:
            return

        self.plan_id_label.setText(str(self.plan.id))
        self.plan_name_label.setText(self.plan.name)
        self.plan_description_label.setText(self.plan.description)
        self.plan_technique_label.setText(self.plan.technique)
        self.plan_intent_label.setText(self.plan.intent)

        # Primary target
        if self.plan.targets and len(self.plan.targets) > 0:
            primary = self.plan.targets[0]
            self.primary_target_label.setText(primary.name)
        else:
            self.primary_target_label.setText("None")

        # Prescription
        if self.plan.prescription:
            self.prescription_label.setText(f"{self.plan.prescription.dose} Gy")
        else:
            self.prescription_label.setText("Not set")

        # Fractions
        if self.plan.prescription:
            self.fractions_label.setText(str(self.plan.prescription.fractions))
        else:
            self.fractions_label.setText("Not set")

        # Date
        if hasattr(self.plan, "creation_date") and self.plan.creation_date:
            self.plan_date_label.setText(self.plan.creation_date.strftime("%Y-%m-%d"))
        else:
            self.plan_date_label.setText("Unknown")


class ReviewPanel(QWidget):
    """
    Main review panel for reviewing and approving treatment plans.

    This panel provides an Eclipse-like interface for reviewing plans,
    managing the approval workflow, and viewing plan details.

    Attributes
    ----------
    current_plan : TreatmentPlan
        Currently selected treatment plan
    current_workflow : PlanApprovalWorkflow
        Approval workflow for the current plan
    plan_manager : PlanApprovalManager
        Manager for plan approval workflows
    """

    def __init__(self, parent=None):
        """
        Initialize the review panel.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget, by default None
        """
        super().__init__(parent)

        self.current_plan = None
        self.current_workflow = None
        self.plan_manager = get_plan_approval_manager()

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with status
        header_layout = QHBoxLayout()
        self.plan_label = QLabel("No plan selected")
        self.plan_label.setFont(QFont("Arial", 12, QFont.Bold))

        self.status_widget = PlanStatusWidget()

        header_layout.addWidget(self.plan_label)
        header_layout.addStretch()
        header_layout.addWidget(QLabel("Status:"))
        header_layout.addWidget(self.status_widget)

        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        header_widget.setStyleSheet("background-color: #f0f0f0; padding: 5px;")

        layout.addWidget(header_widget)

        # Main content splitter
        self.main_splitter = QSplitter(Qt.Horizontal)

        # Left panel - Plan Review Tab Widget
        self.left_panel = QTabWidget()

        # Tab 1: Plan Summary
        self.plan_summary = PlanReviewSummary()
        self.left_panel.addTab(self.plan_summary, "Plan Summary")

        # Tab 2: Dose Review (MPR viewer)
        self.dose_viewer = MPRViewer()
        self.left_panel.addTab(self.dose_viewer, "Dose Review")

        # Tab 3: DVH View
        self.dvh_view = DVHView()
        self.left_panel.addTab(self.dvh_view, "DVH Analysis")

        # Right panel - Approval workflow
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # Approval history
        self.approval_history = PlanApprovalHistory()
        right_layout.addWidget(self.approval_history)

        # Review action panel
        self.action_panel = ReviewActionPanel()
        self.action_panel.action_executed.connect(self._on_action_executed)
        right_layout.addWidget(self.action_panel)

        # Add panels to splitter
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.right_panel)
        self.main_splitter.setSizes([600, 400])

        layout.addWidget(self.main_splitter)

    def set_plan(self, plan: TreatmentPlan) -> None:
        """
        Set the current plan for review.

        Parameters
        ----------
        plan : TreatmentPlan
            Plan to review
        """
        self.current_plan = plan

        if plan:
            self.plan_label.setText(f"Plan: {plan.name}")

            # Get or create approval workflow
            self.current_workflow = self.plan_manager.get_workflow(str(plan.id))
            if not self.current_workflow:
                self.current_workflow = self.plan_manager.create_workflow(
                    str(plan.id), plan.name
                )

            # Update components
            self.status_widget.set_status(self.current_workflow.status)
            self.plan_summary.set_plan(plan)
            self.approval_history.set_workflow(self.current_workflow)
            self.action_panel.set_workflow(self.current_workflow)

            # Update visualizations if available
            if hasattr(plan, "dose") and plan.dose is not None:
                self.dose_viewer.set_patient_data(
                    plan.image, plan.structure_set, plan.dose
                )
                self.dvh_view.set_plan(plan)
        else:
            self.plan_label.setText("No plan selected")
            self.status_widget.set_status(ApprovalStatus.DRAFT)
            self.current_workflow = None
            self.approval_history.set_workflow(None)
            self.action_panel.set_workflow(None)

            # Clear visualizations
            self.dose_viewer.clear()
            self.dvh_view.clear()

    def _on_action_executed(self) -> None:
        """Handle action execution events."""
        if self.current_workflow:
            # Update UI elements
            self.status_widget.set_status(self.current_workflow.status)
            self.approval_history.set_workflow(self.current_workflow)
            self.action_panel.set_workflow(self.current_workflow)
