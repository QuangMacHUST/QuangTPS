import enum
import datetime
from typing import Dict, List, Optional, Any, cast

from quangtps.common.user import User, UserRole

class ApprovalStatus(enum.Enum):
    """Status of treatment plan approval"""
    DRAFT = "draft"               # Initial draft - not ready for review
    PENDING_PHYSICIAN = "pending_physician"  # Ready for physician review
    PENDING_PHYSICIST = "pending_physicist"  # Ready for physicist review
    APPROVED = "approved"         # Fully approved and ready for treatment
    REJECTED = "rejected"         # Rejected - requires revision
    
    @classmethod
    def get_display_name(cls, status: 'ApprovalStatus') -> str:
        """Get user-friendly display name for a status"""
        names = {
            cls.DRAFT: "Draft",
            cls.PENDING_PHYSICIAN: "Pending Physician Review",
            cls.PENDING_PHYSICIST: "Pending Physicist Review",
            cls.APPROVED: "Approved",
            cls.REJECTED: "Rejected"
        }
        return names.get(status, str(status))
    
    @classmethod
    def get_color(cls, status: 'ApprovalStatus') -> str:
        """Get color associated with a status for UI display"""
        colors = {
            cls.DRAFT: "#888888",
            cls.PENDING_PHYSICIAN: "#FFA500",
            cls.PENDING_PHYSICIST: "#3498DB",
            cls.APPROVED: "#2ECC71",
            cls.REJECTED: "#E74C3C"
        }
        return colors.get(status, "#FFFFFF")  # Default white

class ApprovalAction(enum.Enum):
    """Actions that can be taken on a treatment plan"""
    SUBMIT = "submit"             # Submit for review
    APPROVE = "approve"           # Approve the plan
    REJECT = "reject"             # Reject the plan
    REVERT = "revert"             # Revert to draft for changes
    
    @classmethod
    def get_display_name(cls, action: 'ApprovalAction') -> str:
        """Get user-friendly display name for an action"""
        names = {
            cls.SUBMIT: "Submit for Review",
            cls.APPROVE: "Approve",
            cls.REJECT: "Reject",
            cls.REVERT: "Revert to Draft"
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
            "action": self.action.value,
            "user": self.user.username,
            "timestamp": self.timestamp.isoformat(),
            "comment": self.comment,
            "from_status": self.from_status.value,
            "to_status": self.to_status.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], users_map: Dict[str, User]) -> "ApprovalHistoryEntry":
        """Create from dictionary with user mapping"""
        user = users_map.get(data["user"])
        if user is None:
            raise ValueError(f"User {data['user']} not found in users map")
        
        return cls(
            action=ApprovalAction(data["action"]),
            user=user,
            timestamp=datetime.datetime.fromisoformat(data["timestamp"]),
            comment=data["comment"],
            from_status=ApprovalStatus(data["from_status"]),
            to_status=ApprovalStatus(data["to_status"])
        )
        
class ApprovalWorkflow:
    """Manages the approval workflow for treatment plans"""
    
    def __init__(self):
        self.status_transitions: Dict[tuple[ApprovalStatus, ApprovalAction], ApprovalStatus] = {
            # From DRAFT
            (ApprovalStatus.DRAFT, ApprovalAction.SUBMIT): ApprovalStatus.PENDING_PHYSICIAN,
            
            # From PENDING_PHYSICIAN
            (ApprovalStatus.PENDING_PHYSICIAN, ApprovalAction.APPROVE): ApprovalStatus.PENDING_PHYSICIST,
            (ApprovalStatus.PENDING_PHYSICIAN, ApprovalAction.REJECT): ApprovalStatus.REJECTED,
            (ApprovalStatus.PENDING_PHYSICIAN, ApprovalAction.REVERT): ApprovalStatus.DRAFT,
            
            # From PENDING_PHYSICIST
            (ApprovalStatus.PENDING_PHYSICIST, ApprovalAction.APPROVE): ApprovalStatus.APPROVED,
            (ApprovalStatus.PENDING_PHYSICIST, ApprovalAction.REJECT): ApprovalStatus.REJECTED,
            (ApprovalStatus.PENDING_PHYSICIST, ApprovalAction.REVERT): ApprovalStatus.PENDING_PHYSICIAN,
            
            # From REJECTED
            (ApprovalStatus.REJECTED, ApprovalAction.REVERT): ApprovalStatus.DRAFT,
            
            # No actions from APPROVED
        }
        
        # Define which roles can perform which actions on which statuses
        self.role_permissions: Dict[UserRole, Dict[ApprovalStatus, List[ApprovalAction]]] = {
            # Planner permissions
            UserRole.PLANNER: {
                ApprovalStatus.DRAFT: [ApprovalAction.SUBMIT],
                ApprovalStatus.PENDING_PHYSICIAN: [],
                ApprovalStatus.PENDING_PHYSICIST: [],
                ApprovalStatus.APPROVED: [],
                ApprovalStatus.REJECTED: [ApprovalAction.REVERT]
            },
            
            # Physician permissions
            UserRole.PHYSICIAN: {
                ApprovalStatus.DRAFT: [ApprovalAction.SUBMIT],
                ApprovalStatus.PENDING_PHYSICIAN: [ApprovalAction.APPROVE, ApprovalAction.REJECT, ApprovalAction.REVERT],
                ApprovalStatus.PENDING_PHYSICIST: [],
                ApprovalStatus.APPROVED: [],
                ApprovalStatus.REJECTED: [ApprovalAction.REVERT]
            },
            
            # Physicist permissions
            UserRole.PHYSICIST: {
                ApprovalStatus.DRAFT: [],
                ApprovalStatus.PENDING_PHYSICIAN: [],
                ApprovalStatus.PENDING_PHYSICIST: [ApprovalAction.APPROVE, ApprovalAction.REJECT, ApprovalAction.REVERT],
                ApprovalStatus.APPROVED: [],
                ApprovalStatus.REJECTED: []
            },
            
            # Admin permissions (can do everything)
            UserRole.ADMIN: {
                ApprovalStatus.DRAFT: [ApprovalAction.SUBMIT],
                ApprovalStatus.PENDING_PHYSICIAN: [ApprovalAction.APPROVE, ApprovalAction.REJECT, ApprovalAction.REVERT],
                ApprovalStatus.PENDING_PHYSICIST: [ApprovalAction.APPROVE, ApprovalAction.REJECT, ApprovalAction.REVERT],
                ApprovalStatus.APPROVED: [ApprovalAction.REVERT],
                ApprovalStatus.REJECTED: [ApprovalAction.REVERT]
            },
            
            # Guest permissions (view only)
            UserRole.GUEST: {
                ApprovalStatus.DRAFT: [],
                ApprovalStatus.PENDING_PHYSICIAN: [],
                ApprovalStatus.PENDING_PHYSICIST: [],
                ApprovalStatus.APPROVED: [],
                ApprovalStatus.REJECTED: []
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
            "status": self.status.value,
            "history": [entry.to_dict() for entry in self.history],
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "last_modified_by": self.last_modified_by.username if self.last_modified_by else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], users_map: Dict[str, User]) -> "PlanApproval":
        """Create from dictionary with user mapping"""
        approval = cls(data["plan_id"])
        approval.status = ApprovalStatus(data["status"])
        
        approval.history = [
            ApprovalHistoryEntry.from_dict(entry_data, users_map)
            for entry_data in data.get("history", [])
        ]
        
        if data.get("last_modified"):
            approval.last_modified = datetime.datetime.fromisoformat(data["last_modified"])
            
        if data.get("last_modified_by") and data["last_modified_by"] in users_map:
            approval.last_modified_by = users_map[data["last_modified_by"]]
            
        return approval 