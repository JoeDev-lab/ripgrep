"""
Utility functions with various patterns for testing.
"""

from typing import Any


def helper() -> None:
    """A simple helper function."""
    pass


def data_processor(data: Any) -> Any:
    """Process some data."""
    return data


# Short identifier functions (for regex fallback testing)
def a() -> None:
    """A very short function name."""
    pass


def b() -> None:
    """Another short function."""
    pass


def x() -> None:
    """Yet another short function."""
    pass


# Multi-line function (for context padding testing)
def multi_line_function(
    param1: str,
    param2: int,
    param3: float,
    param4: bool,
    param5: dict
) -> tuple:
    """
    A multi-line function with many parameters.
    
    This function tests the context padding feature by having
    more than 15 lines of code.
    """
    # Line 1
    # Line 2
    # Line 3
    # Line 4
    # Line 5
    # Line 6
    # Line 7
    # Line 8
    # Line 9
    # Line 10
    # Line 11
    # Line 12
    # Line 13
    # Line 14
    # Line 15
    # Line 16
    return (param1, param2, param3, param4, param5)
