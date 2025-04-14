#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eclipse-like plan approval workflow system for QuangTPS.

This module implements a comprehensive plan approval workflow similar to
Varian Eclipse TPS, allowing multi-level approval of treatment plans by
different staff roles (physician, physicist, technician, etc.).

All plan changes are tracked with proper versioning and an audit trail
is maintained for each approval or rejection event.
"""

import enum
import logging
import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

logger = logging.getLogger(__name__)

class ApprovalRole(enum.Enum):
    """Roles responsible for plan approval in radiotherapy workflow."""
    PHYSICIAN = "Physician"
    PHYSICIST = "Physicist"
    DOSIMETRIST = "Dosimetrist"
    TECHNICIAN = "Technician" 
    PLANNER = "Planner"
    ADMINISTRATOR = "Administrator"

class ApprovalStatus(enum.Enum):
    """Status options for plan approval workflow."""
    DRAFT = "Draft"
    PLANNING = "Planning"
    PENDING_APPROVAL = "Pending Approval"
    APPROVED_BY_PLANNER = "Approved by Planner"
    APPROVED_BY_PHYSICIST = "Approved by Physicist"
    APPROVED_BY_PHYSICIAN = "Approved by Physician"
    TREATMENT_APPROVED = "Treatment Approved"
    REJECTED = "Rejected"
    UNDER_REVISION = "Under Revision"
    ARCHIVED = "Archived"

class ApprovalAction(enum.Enum):
    """Actions that can be taken during the approval workflow."""
    SUBMIT = "Submit for Approval"
    APPROVE = "Approve"
    REJECT = "Reject"
    RETURN_TO_PLANNING = "Return to Planning"
    ARCHIVE = "Archive"
    RESTORE = "Restore"

class ApprovalEvent:
    """
    Represents a single approval workflow event in a plan's history.
    
    Attributes
    ----------
    timestamp : datetime
        When the event occurred
    user : str
        Username of the person who performed the action
    role : ApprovalRole
        Role of the user who performed the action
    action : ApprovalAction
        Action that was performed
    old_status : ApprovalStatus
        Status before the action
    new_status : ApprovalStatus
        Status after the action
    comment : str
        Optional comment explaining the action
    """
    
    def __init__(
        self,
        user: str,
        role: ApprovalRole,
        action: ApprovalAction,
        old_status: ApprovalStatus,
        new_status: ApprovalStatus,
        comment: str = "",
    ):
        """
        Initialize an ApprovalEvent.
        
        Parameters
        ----------
        user : str
            Username of the person who performed the action
        role : ApprovalRole
            Role of the user who performed the action
        action : ApprovalAction
            Action that was performed
        old_status : ApprovalStatus
            Status before the action
        new_status : ApprovalStatus
            Status after the action
        comment : str, optional
            Optional comment explaining the action, by default ""
        """
        self.timestamp = datetime.datetime.now()
        self.user = user
        self.role = role
        self.action = action
        self.old_status = old_status
        self.new_status = new_status
        self.comment = comment
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the approval event to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the event
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "user": self.user,
            "role": self.role.value,
            "action": self.action.value,
            "old_status": self.old_status.value,
            "new_status": self.new_status.value,
            "comment": self.comment,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApprovalEvent":
        """
        Create an ApprovalEvent from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary representation of the event
            
        Returns
        -------
        ApprovalEvent
            Created approval event
        """
        event = cls(
            user=data["user"],
            role=ApprovalRole(data["role"]),
            action=ApprovalAction(data["action"]),
            old_status=ApprovalStatus(data["old_status"]),
            new_status=ApprovalStatus(data["new_status"]),
            comment=data.get("comment", ""),
        )
        if "timestamp" in data:
            event.timestamp = datetime.datetime.fromisoformat(data["timestamp"])
        
        return event
    
    def __str__(self) -> str:
        """
        Get a string representation of the approval event.
        
        Returns
        -------
        str
            String representation of the event
        """
        return (
            f"{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - "
            f"{self.user} ({self.role.value}) performed {self.action.value}: "
            f"{self.old_status.value} → {self.new_status.value}"
        )

class PlanApprovalWorkflow:
    """
    Manages the Eclipse-like plan approval workflow.
    
    This class handles the approval process for treatment plans,
    enforcing proper workflow transitions, maintaining an audit
    trail, and ensuring only authorized roles can perform specific
    actions.
    
    Attributes
    ----------
    plan_id : str
        Unique identifier for the plan
    status : ApprovalStatus
        Current approval status of the plan
    history : List[ApprovalEvent]
        Chronological list of all approval events
    required_approvals : Dict[ApprovalRole, bool]
        Required approvals and their current state
    comments : List[Tuple[datetime.datetime, str, str]]
        List of comments with timestamp, user, and comment text
    """
    
    def __init__(self, plan_id: str, plan_name: str = ""):
        """
        Initialize a PlanApprovalWorkflow for a given plan.
        
        Parameters
        ----------
        plan_id : str
            Unique identifier for the plan
        plan_name : str, optional
            Name of the plan, by default ""
        """
        self.plan_id = plan_id
        self.plan_name = plan_name
        self.status = ApprovalStatus.DRAFT
        self.history: List[ApprovalEvent] = []
        self.required_approvals: Dict[ApprovalRole, bool] = {
            ApprovalRole.PLANNER: False,
            ApprovalRole.PHYSICIST: False,
            ApprovalRole.PHYSICIAN: False,
        }
        self.comments: List[Tuple[datetime.datetime, str, str]] = []
        
        # Create the initial event
        self._add_event(
            user="System",
            role=ApprovalRole.ADMINISTRATOR,
            action=ApprovalAction.SUBMIT,
            old_status=ApprovalStatus.DRAFT,
            new_status=ApprovalStatus.DRAFT,
            comment="Plan workflow initialized"
        )
    
    def _add_event(
        self,
        user: str,
        role: ApprovalRole,
        action: ApprovalAction,
        old_status: ApprovalStatus,
        new_status: ApprovalStatus,
        comment: str = "",
    ) -> None:
        """
        Add a new approval event to the history.
        
        Parameters
        ----------
        user : str
            Username of the person who performed the action
        role : ApprovalRole
            Role of the user who performed the action
        action : ApprovalAction
            Action that was performed
        old_status : ApprovalStatus
            Status before the action
        new_status : ApprovalStatus
            Status after the action
        comment : str, optional
            Optional comment explaining the action, by default ""
        """
        event = ApprovalEvent(
            user=user,
            role=role,
            action=action,
            old_status=old_status,
            new_status=new_status,
            comment=comment,
        )
        self.history.append(event)
        
        # Add comment if provided
        if comment:
            self.add_comment(user, comment)
    
    def add_comment(self, user: str, comment: str) -> None:
        """
        Add a comment to the plan approval workflow.
        
        Parameters
        ----------
        user : str
            Username of the person adding the comment
        comment : str
            Text of the comment
        """
        self.comments.append((datetime.datetime.now(), user, comment))
    
    def submit_for_approval(self, user: str, role: ApprovalRole, comment: str = "") -> bool:
        """
        Submit the plan for approval.
        
        Parameters
        ----------
        user : str
            Username of the person submitting the plan
        role : ApprovalRole
            Role of the user submitting the plan
        comment : str, optional
            Explanation for the submission, by default ""
            
        Returns
        -------
        bool
            True if successfully submitted, False otherwise
        """
        if self.status in [ApprovalStatus.DRAFT, ApprovalStatus.PLANNING, ApprovalStatus.UNDER_REVISION]:
            old_status = self.status
            self.status = ApprovalStatus.PENDING_APPROVAL
            
            self._add_event(
                user=user,
                role=role,
                action=ApprovalAction.SUBMIT,
                old_status=old_status,
                new_status=self.status,
                comment=comment,
            )
            
            logger.info(f"Plan {self.plan_id} submitted for approval by {user} ({role.value})")
            return True
        else:
            logger.warning(
                f"Cannot submit plan {self.plan_id} for approval from status {self.status.value}"
            )
            return False
    
    def approve(
        self, user: str, role: ApprovalRole, comment: str = ""
    ) -> bool:
        """
        Approve the plan for the given role.
        
        Parameters
        ----------
        user : str
            Username of the person approving the plan
        role : ApprovalRole
            Role of the user approving the plan
        comment : str, optional
            Explanation for the approval, by default ""
            
        Returns
        -------
        bool
            True if successfully approved, False otherwise
        """
        if role not in self.required_approvals:
            logger.warning(f"Role {role.value} not required for plan approval")
            return False
        
        # Only certain statuses can be approved
        valid_statuses = [
            ApprovalStatus.PENDING_APPROVAL,
            ApprovalStatus.APPROVED_BY_PLANNER,
            ApprovalStatus.APPROVED_BY_PHYSICIST,
        ]
        
        if self.status not in valid_statuses:
            logger.warning(
                f"Cannot approve plan {self.plan_id} from status {self.status.value}"
            )
            return False
        
        # Update approval state
        self.required_approvals[role] = True
        
        # Determine the new status based on role
        old_status = self.status
        
        if role == ApprovalRole.PLANNER:
            self.status = ApprovalStatus.APPROVED_BY_PLANNER
        elif role == ApprovalRole.PHYSICIST:
            self.status = ApprovalStatus.APPROVED_BY_PHYSICIST
        elif role == ApprovalRole.PHYSICIAN:
            self.status = ApprovalStatus.APPROVED_BY_PHYSICIAN
            
            # Check if all required approvals are complete
            if all(self.required_approvals.values()):
                self.status = ApprovalStatus.TREATMENT_APPROVED
        
        # Add the event
        self._add_event(
            user=user,
            role=role,
            action=ApprovalAction.APPROVE,
            old_status=old_status,
            new_status=self.status,
            comment=comment,
        )
        
        logger.info(f"Plan {self.plan_id} approved by {user} ({role.value})")
        return True
    
    def reject(
        self, user: str, role: ApprovalRole, comment: str = ""
    ) -> bool:
        """
        Reject the plan and return it to planning.
        
        Parameters
        ----------
        user : str
            Username of the person rejecting the plan
        role : ApprovalRole
            Role of the user rejecting the plan
        comment : str, optional
            Explanation for the rejection, by default ""
            
        Returns
        -------
        bool
            True if successfully rejected, False otherwise
        """
        if self.status in [
            ApprovalStatus.PENDING_APPROVAL,
            ApprovalStatus.APPROVED_BY_PLANNER,
            ApprovalStatus.APPROVED_BY_PHYSICIST,
            ApprovalStatus.APPROVED_BY_PHYSICIAN,
        ]:
            old_status = self.status
            self.status = ApprovalStatus.UNDER_REVISION
            
            # Reset all approvals
            for role_key in self.required_approvals:
                self.required_approvals[role_key] = False
            
            self._add_event(
                user=user,
                role=role,
                action=ApprovalAction.REJECT,
                old_status=old_status,
                new_status=self.status,
                comment=comment,
            )
            
            logger.info(f"Plan {self.plan_id} rejected by {user} ({role.value})")
            return True
        else:
            logger.warning(
                f"Cannot reject plan {self.plan_id} from status {self.status.value}"
            )
            return False
    
    def return_to_planning(
        self, user: str, role: ApprovalRole, comment: str = ""
    ) -> bool:
        """
        Return the plan to planning stage.
        
        Parameters
        ----------
        user : str
            Username of the person returning the plan
        role : ApprovalRole
            Role of the user returning the plan
        comment : str, optional
            Explanation for returning the plan, by default ""
            
        Returns
        -------
        bool
            True if successfully returned, False otherwise
        """
        if self.status in [
            ApprovalStatus.UNDER_REVISION,
            ApprovalStatus.PENDING_APPROVAL,
            ApprovalStatus.REJECTED,
        ]:
            old_status = self.status
            self.status = ApprovalStatus.PLANNING
            
            self._add_event(
                user=user,
                role=role,
                action=ApprovalAction.RETURN_TO_PLANNING,
                old_status=old_status,
                new_status=self.status,
                comment=comment,
            )
            
            logger.info(f"Plan {self.plan_id} returned to planning by {user} ({role.value})")
            return True
        else:
            logger.warning(
                f"Cannot return plan {self.plan_id} to planning from status {self.status.value}"
            )
            return False
    
    def archive(
        self, user: str, role: ApprovalRole, comment: str = ""
    ) -> bool:
        """
        Archive the plan.
        
        Parameters
        ----------
        user : str
            Username of the person archiving the plan
        role : ApprovalRole
            Role of the user archiving the plan
        comment : str, optional
            Explanation for archiving, by default ""
            
        Returns
        -------
        bool
            True if successfully archived, False otherwise
        """
        if role not in [ApprovalRole.ADMINISTRATOR, ApprovalRole.PHYSICIST, ApprovalRole.PHYSICIAN]:
            logger.warning(f"Role {role.value} not authorized to archive plans")
            return False
        
        if self.status != ApprovalStatus.ARCHIVED:
            old_status = self.status
            self.status = ApprovalStatus.ARCHIVED
            
            self._add_event(
                user=user,
                role=role,
                action=ApprovalAction.ARCHIVE,
                old_status=old_status,
                new_status=self.status,
                comment=comment,
            )
            
            logger.info(f"Plan {self.plan_id} archived by {user} ({role.value})")
            return True
        else:
            logger.warning(f"Plan {self.plan_id} is already archived")
            return False
    
    def restore(
        self, user: str, role: ApprovalRole, comment: str = ""
    ) -> bool:
        """
        Restore an archived plan to its previous status.
        
        Parameters
        ----------
        user : str
            Username of the person restoring the plan
        role : ApprovalRole
            Role of the user restoring the plan
        comment : str, optional
            Explanation for restoring, by default ""
            
        Returns
        -------
        bool
            True if successfully restored, False otherwise
        """
        if role not in [ApprovalRole.ADMINISTRATOR, ApprovalRole.PHYSICIST, ApprovalRole.PHYSICIAN]:
            logger.warning(f"Role {role.value} not authorized to restore plans")
            return False
        
        if self.status == ApprovalStatus.ARCHIVED:
            # Find the status before archiving
            for event in reversed(self.history):
                if event.action == ApprovalAction.ARCHIVE:
                    old_status = self.status
                    self.status = event.old_status
                    
                    self._add_event(
                        user=user,
                        role=role,
                        action=ApprovalAction.RESTORE,
                        old_status=old_status,
                        new_status=self.status,
                        comment=comment,
                    )
                    
                    logger.info(f"Plan {self.plan_id} restored by {user} ({role.value})")
                    return True
            
            # If no archive event is found, restore to DRAFT
            old_status = self.status
            self.status = ApprovalStatus.DRAFT
            
            self._add_event(
                user=user,
                role=role,
                action=ApprovalAction.RESTORE,
                old_status=old_status,
                new_status=self.status,
                comment=comment,
            )
            
            logger.info(f"Plan {self.plan_id} restored to draft by {user} ({role.value})")
            return True
        else:
            logger.warning(f"Plan {self.plan_id} is not archived")
            return False
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete history of approval workflow events.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of approval events as dictionaries
        """
        return [event.to_dict() for event in self.history]
    
    def get_comments(self) -> List[Dict[str, Any]]:
        """
        Get all comments made during the approval workflow.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of comments with timestamp, user, and text
        """
        return [
            {"timestamp": timestamp.isoformat(), "user": user, "comment": comment}
            for timestamp, user, comment in self.comments
        ]
    
    def is_fully_approved(self) -> bool:
        """
        Check if the plan is fully approved.
        
        Returns
        -------
        bool
            True if fully approved, False otherwise
        """
        return self.status == ApprovalStatus.TREATMENT_APPROVED
    
    def get_pending_approvals(self) -> List[ApprovalRole]:
        """
        Get a list of roles that still need to approve the plan.
        
        Returns
        -------
        List[ApprovalRole]
            List of roles that haven't approved yet
        """
        return [role for role, approved in self.required_approvals.items() if not approved]
    
    def get_status_display(self) -> Dict[str, Any]:
        """
        Get a dictionary with detailed status information.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with detailed status information
        """
        return {
            "status": self.status.value,
            "approvals": {role.value: approved for role, approved in self.required_approvals.items()},
            "pending": [role.value for role in self.get_pending_approvals()],
            "fully_approved": self.is_fully_approved(),
            "last_modified": self.history[-1].timestamp.isoformat() if self.history else None,
            "version": len(self.history),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the workflow to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the workflow
        """
        return {
            "plan_id": self.plan_id,
            "plan_name": self.plan_name,
            "status": self.status.value,
            "required_approvals": {role.value: approved for role, approved in self.required_approvals.items()},
            "history": self.get_history(),
            "comments": self.get_comments(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanApprovalWorkflow":
        """
        Create a PlanApprovalWorkflow from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary representation of the workflow
            
        Returns
        -------
        PlanApprovalWorkflow
            Created workflow
        """
        workflow = cls(
            plan_id=data["plan_id"],
            plan_name=data.get("plan_name", ""),
        )
        
        workflow.status = ApprovalStatus(data["status"])
        workflow.required_approvals = {
            ApprovalRole(role): approved
            for role, approved in data.get("required_approvals", {}).items()
        }
        
        # Clear default history
        workflow.history = []
        
        # Add history events
        for event_data in data.get("history", []):
            workflow.history.append(ApprovalEvent.from_dict(event_data))
        
        # Add comments
        workflow.comments = []
        for comment_data in data.get("comments", []):
            timestamp = datetime.datetime.fromisoformat(comment_data["timestamp"])
            user = comment_data["user"]
            comment = comment_data["comment"]
            workflow.comments.append((timestamp, user, comment))
        
        return workflow
    
    def __str__(self) -> str:
        """
        Get a string representation of the workflow.
        
        Returns
        -------
        str
            String representation of the workflow
        """
        return (
            f"Plan '{self.plan_name}' (ID: {self.plan_id}): "
            f"Status: {self.status.value}, "
            f"Pending approvals: {[role.value for role in self.get_pending_approvals()]}"
        )

class PlanApprovalManager:
    """
    Manager for handling the approval processes of multiple plans.
    
    This class manages PlanApprovalWorkflow instances for multiple plans,
    provides queries for plans in various states, and handles persistence.
    
    Attributes
    ----------
    workflows : Dict[str, PlanApprovalWorkflow]
        Dictionary of workflows indexed by plan_id
    """
    
    def __init__(self):
        """Initialize an empty PlanApprovalManager."""
        self.workflows: Dict[str, PlanApprovalWorkflow] = {}
    
    def create_workflow(self, plan_id: str, plan_name: str = "") -> PlanApprovalWorkflow:
        """
        Create a new approval workflow for a plan.
        
        Parameters
        ----------
        plan_id : str
            Unique identifier for the plan
        plan_name : str, optional
            Name of the plan, by default ""
            
        Returns
        -------
        PlanApprovalWorkflow
            The created workflow
            
        Raises
        ------
        ValueError
            If a workflow already exists for the plan
        """
        if plan_id in self.workflows:
            raise ValueError(f"Approval workflow already exists for plan {plan_id}")
        
        workflow = PlanApprovalWorkflow(plan_id, plan_name)
        self.workflows[plan_id] = workflow
        
        logger.info(f"Created approval workflow for plan {plan_id}")
        return workflow
    
    def get_workflow(self, plan_id: str) -> Optional[PlanApprovalWorkflow]:
        """
        Get the approval workflow for a plan.
        
        Parameters
        ----------
        plan_id : str
            Unique identifier for the plan
            
        Returns
        -------
        Optional[PlanApprovalWorkflow]
            The workflow if it exists, None otherwise
        """
        return self.workflows.get(plan_id)
    
    def delete_workflow(self, plan_id: str) -> bool:
        """
        Delete the approval workflow for a plan.
        
        Parameters
        ----------
        plan_id : str
            Unique identifier for the plan
            
        Returns
        -------
        bool
            True if deleted, False if not found
        """
        if plan_id in self.workflows:
            del self.workflows[plan_id]
            logger.info(f"Deleted approval workflow for plan {plan_id}")
            return True
        return False
    
    def get_plans_by_status(self, status: ApprovalStatus) -> List[PlanApprovalWorkflow]:
        """
        Get all plans with a particular status.
        
        Parameters
        ----------
        status : ApprovalStatus
            Status to filter by
            
        Returns
        -------
        List[PlanApprovalWorkflow]
            List of workflows with the specified status
        """
        return [
            workflow for workflow in self.workflows.values() if workflow.status == status
        ]
    
    def get_plans_pending_approval(
        self, role: Optional[ApprovalRole] = None
    ) -> List[PlanApprovalWorkflow]:
        """
        Get all plans pending approval, optionally filtered by role.
        
        Parameters
        ----------
        role : Optional[ApprovalRole], optional
            Role to filter by, by default None
            
        Returns
        -------
        List[PlanApprovalWorkflow]
            List of workflows pending approval
        """
        if role is None:
            # Get all plans with pending approvals
            return [
                workflow
                for workflow in self.workflows.values()
                if workflow.status in [
                    ApprovalStatus.PENDING_APPROVAL,
                    ApprovalStatus.APPROVED_BY_PLANNER,
                    ApprovalStatus.APPROVED_BY_PHYSICIST,
                ]
                and not workflow.is_fully_approved()
            ]
        else:
            # Get plans pending approval for a specific role
            return [
                workflow
                for workflow in self.workflows.values()
                if not workflow.is_fully_approved()
                and role in workflow.get_pending_approvals()
            ]
    
    def get_approved_plans(self) -> List[PlanApprovalWorkflow]:
        """
        Get all fully approved plans.
        
        Returns
        -------
        List[PlanApprovalWorkflow]
            List of fully approved workflows
        """
        return [
            workflow
            for workflow in self.workflows.values()
            if workflow.is_fully_approved()
        ]
    
    def get_plans_in_planning(self) -> List[PlanApprovalWorkflow]:
        """
        Get all plans in the planning stage.
        
        Returns
        -------
        List[PlanApprovalWorkflow]
            List of workflows in planning
        """
        return [
            workflow
            for workflow in self.workflows.values()
            if workflow.status in [
                ApprovalStatus.DRAFT,
                ApprovalStatus.PLANNING,
                ApprovalStatus.UNDER_REVISION,
            ]
        ]
    
    def get_archived_plans(self) -> List[PlanApprovalWorkflow]:
        """
        Get all archived plans.
        
        Returns
        -------
        List[PlanApprovalWorkflow]
            List of archived workflows
        """
        return self.get_plans_by_status(ApprovalStatus.ARCHIVED)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the manager to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the manager
        """
        return {
            "workflows": {
                plan_id: workflow.to_dict()
                for plan_id, workflow in self.workflows.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanApprovalManager":
        """
        Create a PlanApprovalManager from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary representation of the manager
            
        Returns
        -------
        PlanApprovalManager
            Created manager
        """
        manager = cls()
        
        for plan_id, workflow_data in data.get("workflows", {}).items():
            manager.workflows[plan_id] = PlanApprovalWorkflow.from_dict(workflow_data)
        
        return manager
    
    def save_to_database(self, db_connector) -> bool:
        """
        Save all workflows to the database.
        
        Parameters
        ----------
        db_connector
            Database connector object
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            # This is a placeholder - implementation depends on database structure
            data = self.to_dict()
            # db_connector.save_plan_approvals(data)
            return True
        except Exception as e:
            logger.error(f"Error saving plan approvals to database: {e}")
            return False
    
    @classmethod
    def load_from_database(cls, db_connector) -> "PlanApprovalManager":
        """
        Load all workflows from the database.
        
        Parameters
        ----------
        db_connector
            Database connector object
            
        Returns
        -------
        PlanApprovalManager
            Loaded manager
        """
        try:
            # This is a placeholder - implementation depends on database structure
            # data = db_connector.load_plan_approvals()
            # return cls.from_dict(data)
            return cls()
        except Exception as e:
            logger.error(f"Error loading plan approvals from database: {e}")
            return cls()

# Create a singleton instance for global access
_plan_approval_manager: Optional[PlanApprovalManager] = None

def get_plan_approval_manager() -> PlanApprovalManager:
    """
    Get the singleton instance of PlanApprovalManager.
    
    Returns
    -------
    PlanApprovalManager
        The singleton instance
    """
    global _plan_approval_manager
    if _plan_approval_manager is None:
        _plan_approval_manager = PlanApprovalManager()
    return _plan_approval_manager 