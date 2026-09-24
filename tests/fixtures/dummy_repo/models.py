"""
Model definitions for testing.
"""

from typing import Optional


class UserModel:
    """
    User model class with methods.
    
    This class should be found by class name queries.
    """
    
    def __init__(self, user_id: int, username: str):
        """Initialize the user model."""
        self.user_id = user_id
        self.username = username
    
    def get_user_info(self) -> dict:
        """Get user information."""
        return {
            "user_id": self.user_id,
            "username": self.username
        }
    
    def update_username(self, new_username: str) -> bool:
        """Update the user's username."""
        if new_username:
            self.username = new_username
            return True
        return False
