from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import uuid

from quangtps.planning.approval import PlanApproval, ApprovalStatus, ApprovalAction
from quangtps.planning.evaluation import PlanEvaluation
from quangtps.common.user import User, UserRole

class TreatmentPlan:
    """Treatment plan for a patient"""
    
    def __init__(self, plan_id: Optional[str] = None):
        self.id = plan_id if plan_id else str(uuid.uuid4())
        self.name = "New Plan"
        self.description = ""
        self.created_date = datetime.now()
        self.last_modified = datetime.now()
        self.created_by = ""
        self.approval = PlanApproval(self.id)
        self.evaluation: Optional[PlanEvaluation] = None
        # Other plan properties will be added here
    
    @property
    def approval_status(self) -> ApprovalStatus:
        """Get the current approval status of the plan"""
        return self.approval.status
    
    def get_approval_status_display(self) -> str:
        """Get a user-friendly display string for the approval status"""
        return ApprovalStatus.get_display_name(self.approval_status)
    
    def get_approval_status_color(self) -> str:
        """Get the color code for the approval status"""
        return ApprovalStatus.get_color(self.approval_status)
    
    def get_available_approval_actions(self, user: User) -> List[ApprovalAction]:
        """Get list of available approval actions for the current user"""
        return self.approval.get_available_actions(user)
    
    def perform_approval_action(
        self, 
        action: ApprovalAction, 
        user: User,
        comment: str = ""
    ) -> bool:
        """Perform an approval workflow action"""
        success = self.approval.perform_action(action, user, comment)
        if success:
            self.last_modified = datetime.now()
        return success
    
    def is_editable(self, user: User) -> bool:
        """Check if plan is editable based on approval status and user role"""
        if self.approval_status == ApprovalStatus.DRAFT:
            return True
        if self.approval_status == ApprovalStatus.REJECTED and user.role in [UserRole.PLANNER, UserRole.ADMIN]:
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary for serialization"""
        plan_dict = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_date": self.created_date.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "created_by": self.created_by,
            "approval": self.approval.to_dict(),
        }
        
        if self.evaluation:
            plan_dict["evaluation"] = self.evaluation.to_dict()
            
        return plan_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], users_map: Dict[str, User]) -> 'TreatmentPlan':
        """Create from dictionary with user objects"""
        plan = cls(plan_id=data["id"])
        plan.name = data["name"]
        plan.description = data.get("description", "")
        plan.created_date = datetime.fromisoformat(data["created_date"])
        plan.last_modified = datetime.fromisoformat(data["last_modified"])
        plan.created_by = data["created_by"]
        
        # Load approval data
        if "approval" in data:
            plan.approval = PlanApproval.from_dict(data["approval"], users_map)
        
        # Load evaluation data if present
        if "evaluation" in data and data["evaluation"]:
            from quangtps.planning.evaluation import PlanEvaluation
            plan.evaluation = PlanEvaluation.from_dict(data["evaluation"])
            
        return plan 