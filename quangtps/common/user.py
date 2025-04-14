import os
import json
import enum
import datetime
from typing import Dict, List, Optional, Any, Union

class UserRole(enum.Enum):
    """User roles in the system with different permissions"""
    PLANNER = "planner"       # Treatment planner/dosimetrist
    PHYSICIAN = "physician"   # Physician who approves plans
    PHYSICIST = "physicist"   # Medical physicist who validates plans
    ADMIN = "admin"           # Administrator with full system access
    GUEST = "guest"           # Limited view-only access

class User:
    """User model representing a system user"""
    
    def __init__(self, 
                 username: str, 
                 full_name: str, 
                 email: str,
                 role: UserRole = UserRole.GUEST,
                 is_active: bool = True):
        self.username = username
        self.full_name = full_name
        self.email = email
        self.role = role
        self.is_active = is_active
        self.last_login = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert User to dictionary for serialization"""
        return {
            "username": self.username,
            "full_name": self.full_name,
            "email": self.email,
            "role": self.role.value,
            "is_active": self.is_active,
            "last_login": self.last_login.isoformat() if self.last_login else None
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        """Create User from dictionary"""
        user = cls(
            username=data["username"],
            full_name=data["full_name"],
            email=data["email"],
            role=UserRole(data["role"]),
            is_active=data["is_active"]
        )
        
        if data.get("last_login"):
            user.last_login = datetime.datetime.fromisoformat(data["last_login"])
            
        return user

# Current user information (simple in-memory storage for now)
_current_user = None

def login(username: str, password: str) -> bool:
    """
    Authenticate user with username and password
    
    In a real implementation, this would verify credentials against a secure database
    For this demo, we'll just simulate successful login for any non-empty password
    """
    global _current_user
    
    # In a real implementation, load user data from database and verify password
    # This is just a simplified example
    if username and password:
        user_data = _load_user_data(username)
        if user_data:
            _current_user = User.from_dict(user_data)
            _current_user.last_login = datetime.datetime.now()
            return True
    
    return False

def logout() -> None:
    """Log out the current user"""
    global _current_user
    _current_user = None

def get_current_user() -> Optional[User]:
    """Get the currently logged in user"""
    return _current_user

def set_current_user(user: User) -> None:
    """Set the current user (for testing purposes)"""
    global _current_user
    _current_user = user

def _load_user_data(username: str) -> Optional[Dict[str, Any]]:
    """
    Load user data from storage
    
    In a real implementation, this would query a database
    For this demo, we'll use some hardcoded test users
    """
    # Sample users (in a real system, this would come from a database)
    test_users = {
        "planner": {
            "username": "planner",
            "full_name": "Test Planner",
            "email": "planner@example.com",
            "role": "planner",
            "is_active": True,
            "last_login": None
        },
        "physician": {
            "username": "physician",
            "full_name": "Dr. Test Physician",
            "email": "physician@example.com",
            "role": "physician",
            "is_active": True,
            "last_login": None
        },
        "physicist": {
            "username": "physicist",
            "full_name": "Test Physicist",
            "email": "physicist@example.com",
            "role": "physicist",
            "is_active": True,
            "last_login": None
        },
        "admin": {
            "username": "admin",
            "full_name": "System Administrator",
            "email": "admin@example.com",
            "role": "admin", 
            "is_active": True,
            "last_login": None
        }
    }
    
    return test_users.get(username) 