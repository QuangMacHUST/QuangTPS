import enum
import datetime
from typing import Dict, List, Optional, Any, cast

# Import ApprovalStatus and ApprovalAction directly from enum module to avoid circular imports
from enum import Enum, auto
from quangtps.common.user import User, UserRole

# Use the same enums from core.patient to ensure consistency
class ApprovalStatus(Enum):
    """Enum representing the approval status of a treatment plan."""
    DRAFT = auto()      # Initial state, plan is being created/modified
    PENDING = auto()    # Submitted for review/approval
    APPROVED = auto()   # Fully approved and ready for treatment
    REJECTED = auto()   # Rejected, requires modifications
    DELIVERED = auto()  # Plan has been delivered to patient
    ARCHIVED = auto()   # Plan is archived and no longer active

class ApprovalAction(Enum):
    """Enum representing actions that can be taken in the approval workflow."""
    CREATE = auto()     # Plan creation
    MODIFY = auto()     # Plan modification
    SUBMIT = auto()     # Submit for approval
    APPROVE = auto()    # Approve the plan
    REJECT = auto()     # Reject the plan
    ARCHIVE = auto()    # Archive the plan
    RESTORE = auto()    # Restore from archive

# Extended functionality for ApprovalStatus
class ApprovalStatusExtensions:
    """Extension methods for the ApprovalStatus enum from core."""
    
    @classmethod
    def get_display_name(cls, status: ApprovalStatus) -> str:
        """Get user-friendly display name for a status"""
        names = {
            ApprovalStatus.DRAFT: "Draft",
            ApprovalStatus.PENDING: "Pending Review",
            ApprovalStatus.APPROVED: "Approved",
            ApprovalStatus.REJECTED: "Rejected",
            ApprovalStatus.DELIVERED: "Delivered",
            ApprovalStatus.ARCHIVED: "Archived"
        }
        return names.get(status, str(status))
    
    @classmethod
    def get_color(cls, status: ApprovalStatus) -> str:
        """Get color associated with a status for UI display"""
        colors = {
            ApprovalStatus.DRAFT: "#888888",
            ApprovalStatus.PENDING: "#3498DB",
            ApprovalStatus.APPROVED: "#2ECC71",
            ApprovalStatus.REJECTED: "#E74C3C",
            ApprovalStatus.DELIVERED: "#1ABC9C",
            ApprovalStatus.ARCHIVED: "#7F8C8D"
        }
        return colors.get(status, "#FFFFFF")  # Default white

# Extend the core ApprovalAction with display names
class ApprovalActionExtensions:
    """Extension methods for the ApprovalAction enum from core."""
    
    @classmethod
    def get_display_name(cls, action: ApprovalAction) -> str:
        """Get user-friendly display name for an action"""
        names = {
            ApprovalAction.CREATE: "Create",
            ApprovalAction.MODIFY: "Modify",
            ApprovalAction.SUBMIT: "Submit for Review",
            ApprovalAction.APPROVE: "Approve",
            ApprovalAction.REJECT: "Reject",
            ApprovalAction.ARCHIVE: "Archive",
            ApprovalAction.RESTORE: "Restore"
        }
        return names.get(action, str(action))

class ApprovalHistoryEntry:
    """Record of an approval action taken on a treatment plan"""
    
    def __init__(self,
                 action: ApprovalAction,
                 user: User,
                 timestamp: datetime.datetime,
                 comment: str,
                 from_status: ApprovalStatus,
                 to_status: ApprovalStatus):
        self.action = action
        self.user = user
        self.timestamp = timestamp
        self.comment = comment
        self.from_status = from_status
        self.to_status = to_status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "action": self.action.name,
            "user": self.user.username,
            "timestamp": self.timestamp.isoformat(),
            "comment": self.comment,
            "from_status": self.from_status.name,
            "to_status": self.to_status.name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], users_map: Dict[str, User]) -> "ApprovalHistoryEntry":
        """Create from dictionary with user mapping"""
        user = users_map.get(data["user"])
        if user is None:
            raise ValueError(f"User {data['user']} not found in users map")
        
        return cls(
            action=ApprovalAction[data["action"]],
            user=user,
            timestamp=datetime.datetime.fromisoformat(data["timestamp"]),
            comment=data["comment"],
            from_status=ApprovalStatus[data["from_status"]],
            to_status=ApprovalStatus[data["to_status"]]
        )
        
class ApprovalWorkflow:
    """Manages the approval workflow for treatment plans"""
    
    def __init__(self):
        self.status_transitions: Dict[tuple[ApprovalStatus, ApprovalAction], ApprovalStatus] = {
            # From DRAFT
            (ApprovalStatus.DRAFT, ApprovalAction.SUBMIT): ApprovalStatus.PENDING,
            
            # From PENDING
            (ApprovalStatus.PENDING, ApprovalAction.APPROVE): ApprovalStatus.APPROVED,
            (ApprovalStatus.PENDING, ApprovalAction.REJECT): ApprovalStatus.REJECTED,
            (ApprovalStatus.PENDING, ApprovalAction.MODIFY): ApprovalStatus.DRAFT,
            
            # From REJECTED
            (ApprovalStatus.REJECTED, ApprovalAction.RESTORE): ApprovalStatus.DRAFT,
            
            # From APPROVED
            (ApprovalStatus.APPROVED, ApprovalAction.ARCHIVE): ApprovalStatus.ARCHIVED,
            
            # From ARCHIVED
            (ApprovalStatus.ARCHIVED, ApprovalAction.RESTORE): ApprovalStatus.DRAFT,
        }
        
        # Define which roles can perform which actions on which statuses
        self.role_permissions: Dict[UserRole, Dict[ApprovalStatus, List[ApprovalAction]]] = {
            # Planner permissions
            UserRole.PLANNER: {
                ApprovalStatus.DRAFT: [ApprovalAction.SUBMIT, ApprovalAction.MODIFY],
                ApprovalStatus.PENDING: [],
                ApprovalStatus.APPROVED: [],
                ApprovalStatus.REJECTED: [ApprovalAction.RESTORE],
                ApprovalStatus.DELIVERED: [],
                ApprovalStatus.ARCHIVED: []
            },
            
            # Physician permissions
            UserRole.PHYSICIAN: {
                ApprovalStatus.DRAFT: [ApprovalAction.SUBMIT],
                ApprovalStatus.PENDING: [ApprovalAction.APPROVE, ApprovalAction.REJECT, ApprovalAction.MODIFY],
                ApprovalStatus.APPROVED: [],
                ApprovalStatus.REJECTED: [ApprovalAction.RESTORE],
                ApprovalStatus.DELIVERED: [],
                ApprovalStatus.ARCHIVED: []
            },
            
            # Physicist permissions
            UserRole.PHYSICIST: {
                ApprovalStatus.DRAFT: [],
                ApprovalStatus.PENDING: [ApprovalAction.APPROVE, ApprovalAction.REJECT],
                ApprovalStatus.APPROVED: [ApprovalAction.ARCHIVE],
                ApprovalStatus.REJECTED: [],
                ApprovalStatus.DELIVERED: [],
                ApprovalStatus.ARCHIVED: [ApprovalAction.RESTORE]
            },
            
            # Admin permissions (can do everything)
            UserRole.ADMIN: {
                ApprovalStatus.DRAFT: [ApprovalAction.SUBMIT, ApprovalAction.MODIFY],
                ApprovalStatus.PENDING: [ApprovalAction.APPROVE, ApprovalAction.REJECT, ApprovalAction.MODIFY],
                ApprovalStatus.APPROVED: [ApprovalAction.ARCHIVE, ApprovalAction.MODIFY],
                ApprovalStatus.REJECTED: [ApprovalAction.RESTORE],
                ApprovalStatus.DELIVERED: [ApprovalAction.ARCHIVE],
                ApprovalStatus.ARCHIVED: [ApprovalAction.RESTORE]
            },
            
            # Guest permissions (view only)
            UserRole.GUEST: {
                ApprovalStatus.DRAFT: [],
                ApprovalStatus.PENDING: [],
                ApprovalStatus.APPROVED: [],
                ApprovalStatus.REJECTED: [],
                ApprovalStatus.DELIVERED: [],
                ApprovalStatus.ARCHIVED: []
            }
        }
    
    def get_allowed_actions(self, status: ApprovalStatus, role: UserRole) -> List[ApprovalAction]:
        """Get list of actions allowed for a user role on a plan with given status"""
        role_dict = self.role_permissions.get(role, {})
        return role_dict.get(status, [])
    
    def get_next_status(self, current_status: ApprovalStatus, action: ApprovalAction) -> Optional[ApprovalStatus]:
        """Get the next status after performing an action on a plan with the current status"""
        return self.status_transitions.get((current_status, action))
    
    def can_perform_action(self, current_status: ApprovalStatus, action: ApprovalAction, role: UserRole) -> bool:
        """Check if a user with the given role can perform an action on a plan with the current status"""
        allowed_actions = self.get_allowed_actions(current_status, role)
        return action in allowed_actions

class PlanApproval:
    """Manages the approval status and history for a treatment plan"""
    
    def __init__(self, plan_id: str):
        self.plan_id = plan_id
        self.status = ApprovalStatus.DRAFT
        self.history: List[ApprovalHistoryEntry] = []
        self.workflow = ApprovalWorkflow()
        self.last_modified = datetime.datetime.now()
        self.last_modified_by: Optional[User] = None
    
    def perform_action(self, action: ApprovalAction, user: User, comment: str = "") -> bool:
        """
        Perform an approval action on the plan
        
        Args:
            action: The action to perform
            user: The user performing the action
            comment: Optional comment explaining the action
            
        Returns:
            True if the action was successful, False otherwise
        """
        # Check if user can perform this action
        if not self.workflow.can_perform_action(self.status, action, user.role):
            return False
        
        # Get the next status after this action
        next_status = self.workflow.get_next_status(self.status, action)
        if next_status is None:
            return False
        
        # Record the action in history
        entry = ApprovalHistoryEntry(
            action=action,
            user=user,
            timestamp=datetime.datetime.now(),
            comment=comment,
            from_status=self.status,
            to_status=next_status
        )
        self.history.append(entry)
        
        # Update the status
        self.status = next_status
        self.last_modified = entry.timestamp
        self.last_modified_by = user
        
        return True
    
    def get_available_actions(self, user: User) -> List[ApprovalAction]:
        """Get actions that can be performed by the given user"""
        return self.workflow.get_allowed_actions(self.status, user.role)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "plan_id": self.plan_id,
            "status": self.status.name,
            "history": [entry.to_dict() for entry in self.history],
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "last_modified_by": self.last_modified_by.username if self.last_modified_by else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], users_map: Dict[str, User]) -> "PlanApproval":
        """Create from dictionary with user mapping"""
        approval = cls(data["plan_id"])
        approval.status = ApprovalStatus[data["status"]]
        
        approval.history = [
            ApprovalHistoryEntry.from_dict(entry_data, users_map)
            for entry_data in data.get("history", [])
        ]
        
        if data.get("last_modified"):
            approval.last_modified = datetime.datetime.fromisoformat(data["last_modified"])
            
        if data.get("last_modified_by") and data["last_modified_by"] in users_map:
            approval.last_modified_by = users_map[data["last_modified_by"]]
            
        return approval 

# Export symbols to maintain compatibility with code that imports from core.patient
__all__ = ['ApprovalStatus', 'ApprovalAction', 'ApprovalHistoryEntry', 
           'ApprovalWorkflow', 'PlanApproval', 'ApprovalStatusExtensions', 
           'ApprovalActionExtensions'] 