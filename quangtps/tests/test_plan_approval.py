"""
Tests for the plan approval workflow functionality.
This includes tests for the approval process, status transitions, and UI components.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the parent directory to the path to import the required modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

from ui.plan_approval_dialog import PlanApprovalDialog, PlanStatusIndicator, ApprovalRoleWidget
from core.patient import Patient, TreatmentPlan, ApprovalStatus, ApprovalAction


class TestPlanApprovalWorkflow(unittest.TestCase):
    """Test cases for the plan approval workflow."""
    
    @classmethod
    def setUpClass(cls):
        """Create a QApplication instance for all tests."""
        cls.app = QApplication.instance() or QApplication([])
        
    def setUp(self):
        """Set up test environment before each test."""
        # Create mock patient and plan
        self.patient = MagicMock(spec=Patient)
        self.plan = MagicMock(spec=TreatmentPlan)
        self.plan.name = "Test Plan"
        self.plan.approval_status = ApprovalStatus.DRAFT
        self.plan.approval_history = []
        
        # Create the dialog with the mock plan
        self.dialog = PlanApprovalDialog(self.plan, self.patient)
        
    def test_status_indicator_colors(self):
        """Test that the status indicator shows the correct color for each status."""
        indicator = PlanStatusIndicator()
        
        # Test draft status (yellow)
        indicator.update_status(ApprovalStatus.DRAFT)
        self.assertEqual(indicator.color, Qt.yellow)
        
        # Test approved status (green)
        indicator.update_status(ApprovalStatus.APPROVED)
        self.assertEqual(indicator.color, Qt.green)
        
        # Test rejected status (red)
        indicator.update_status(ApprovalStatus.REJECTED)
        self.assertEqual(indicator.color, Qt.red)
        
    def test_approval_workflow_transitions(self):
        """Test that the approval workflow transitions work correctly."""
        # Initial state should be DRAFT
        self.assertEqual(self.dialog.current_status, ApprovalStatus.DRAFT)
        
        # Mock the user as a physician
        self.dialog.current_user_role = "Physician"
        
        # Submit for approval (transition from DRAFT to PENDING)
        with patch.object(self.plan, 'update_approval_status') as mock_update:
            self.dialog.on_submit_clicked()
            mock_update.assert_called_once_with(
                ApprovalStatus.PENDING, 
                ApprovalAction.SUBMIT, 
                "Physician", 
                ""
            )
            
        # Update the dialog status to PENDING
        self.dialog.update_status(ApprovalStatus.PENDING)
        self.assertEqual(self.dialog.current_status, ApprovalStatus.PENDING)
        
        # Approve the plan (transition from PENDING to APPROVED)
        with patch.object(self.plan, 'update_approval_status') as mock_update:
            self.dialog.on_approve_clicked()
            mock_update.assert_called_once_with(
                ApprovalStatus.APPROVED, 
                ApprovalAction.APPROVE, 
                "Physician",
                ""
            )
    
    def test_role_based_permissions(self):
        """Test that different roles have appropriate permissions."""
        # Test planner role (can only submit)
        self.dialog.current_user_role = "Planner"
        self.dialog.update_button_states()
        self.assertTrue(self.dialog.submit_button.isEnabled())
        self.assertFalse(self.dialog.approve_button.isEnabled())
        
        # Test physician role (can approve and reject)
        self.dialog.current_user_role = "Physician"
        self.dialog.update_status(ApprovalStatus.PENDING)  # Set status to PENDING
        self.dialog.update_button_states()
        self.assertFalse(self.dialog.submit_button.isEnabled())
        self.assertTrue(self.dialog.approve_button.isEnabled())
        self.assertTrue(self.dialog.reject_button.isEnabled())
        
    def test_approval_history_display(self):
        """Test that approval history is displayed correctly."""
        # Setup mock approval history
        history_entries = [
            {"status": ApprovalStatus.DRAFT, "action": ApprovalAction.CREATE, "user": "Planner", "comment": "Initial plan", "timestamp": "2024-07-15 10:00:00"},
            {"status": ApprovalStatus.PENDING, "action": ApprovalAction.SUBMIT, "user": "Planner", "comment": "Ready for review", "timestamp": "2024-07-15 11:30:00"},
            {"status": ApprovalStatus.APPROVED, "action": ApprovalAction.APPROVE, "user": "Physician", "comment": "Approved as planned", "timestamp": "2024-07-15 14:45:00"}
        ]
        
        self.plan.approval_history = history_entries
        
        # Create a new dialog with updated history
        dialog = PlanApprovalDialog(self.plan, self.patient)
        
        # Check that history widget has the correct number of entries
        self.assertEqual(dialog.history_widget.count(), len(history_entries))
        
        # Check content of the first history entry
        first_item_text = dialog.history_widget.item(0).text()
        self.assertIn("Planner", first_item_text)
        self.assertIn("Initial plan", first_item_text)
        
        # Check content of the last history entry
        last_item_text = dialog.history_widget.item(2).text()
        self.assertIn("Physician", last_item_text)
        self.assertIn("Approved", last_item_text)

if __name__ == "__main__":
    unittest.main() 