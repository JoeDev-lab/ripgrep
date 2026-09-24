"""
A function in a deeply nested directory structure.
Tests recursive directory traversal.
"""

from typing import Optional


def deeply_nested_function() -> Optional[str]:
    """
    A function in a deeply nested directory.
    
    This tests that ripgrep can find functions in deeply
    nested directory structures.
    """
    return "test_value"
