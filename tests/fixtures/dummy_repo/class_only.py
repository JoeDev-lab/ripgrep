"""
A class-only file with no functions to test class detection.
"""

from typing import Optional


class ClassOnly:
    """
    A class with only methods, no functions.
    
    This tests that classes are detected even when there
    are no top-level functions.
    """
    
    def method1(self) -> None:
        """First method."""
        pass
    
    def method2(self) -> None:
        """Second method."""
        pass
