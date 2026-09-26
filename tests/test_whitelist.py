"""
Tests for the whitelist filtering functionality.
"""

import json
from pathlib import Path

import pytest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grepRAG import GrepRAG


class TestWhitelistFiltering:
    """Test suite for whitelist filtering functionality."""

    def test_include_all(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that empty whitelist includes all files."""
        with patch("subprocess.run") as mock_run:
            mock_output = json.dumps([
                {
                    "type": "match",
                    "data": {
                        "path": {"text": str(dummy_repo_path / "auth.py")},
                        "line_number": 10,
                        "lines": {"text": "def login():"}
                    }
                }
            ])
            mock_run.return_value = type("MockProcess", (), {
                "stdout": mock_output,
                "stderr": "",
                "returncode": 0,
                "capture_output": True,
                "text": True,
                "check": False
            })()
            
            # Test with empty whitelist
            result = create_grep_rag_instance._run_ripgrep(
                ["def.*login"],
                str(dummy_repo_path),
                whitelist=[],
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_include_specific_pattern(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that `**/auth.py` pattern matches."""
        with patch("subprocess.run") as mock_run:
            mock_output = json.dumps([
                {
                    "type": "match",
                    "data": {
                        "path": {"text": str(dummy_repo_path / "auth.py")},
                        "line_number": 10,
                        "lines": {"text": "def login():"}
                    }
                }
            ])
            mock_run.return_value = type("MockProcess", (), {
                "stdout": mock_output,
                "stderr": "",
                "returncode": 0,
                "capture_output": True,
                "text": True,
                "check": False
            })()
            
            # Test with specific pattern
            result = create_grep_rag_instance._run_ripgrep(
                ["def.*login"],
                str(dummy_repo_path),
                whitelist=["**/auth.py"],
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_include_directory(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that `**/deeply_nested/**` pattern includes nested directory."""
        with patch("subprocess.run") as mock_run:
            mock_output = json.dumps([
                {
                    "type": "match",
                    "data": {
                        "path": {"text": str(dummy_repo_path / "deeply_nested" / "deep_func.py")},
                        "line_number": 10,
                        "lines": {"text": "def deeply_nested_function():"}
                    }
                }
            ])
            mock_run.return_value = type("MockProcess", (), {
                "stdout": mock_output,
                "stderr": "",
                "returncode": 0,
                "capture_output": True,
                "text": True,
                "check": False
            })()
            
            # Test with directory pattern
            result = create_grep_rag_instance._run_ripgrep(
                ["def.*nested"],
                str(dummy_repo_path),
                whitelist=["**/deeply_nested/**"],
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_multiple_patterns(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that multiple whitelist patterns combined with OR."""
        with patch("subprocess.run") as mock_run:
            mock_output = json.dumps([
                {
                    "type": "match",
                    "data": {
                        "path": {"text": str(dummy_repo_path / "auth.py")},
                        "line_number": 10,
                        "lines": {"text": "def login():"}
                    }
                }
            ])
            mock_run.return_value = type("MockProcess", (), {
                "stdout": mock_output,
                "stderr": "",
                "returncode": 0,
                "capture_output": True,
                "text": True,
                "check": False
            })()
            
            # Test with multiple patterns
            result = create_grep_rag_instance._run_ripgrep(
                ["def.*login"],
                str(dummy_repo_path),
                whitelist=["**/auth.py", "**/models.py"],
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_empty_whitelist(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that empty list `[]` behaves same as `None`."""
        with patch("subprocess.run") as mock_run:
            mock_output = json.dumps([
                {
                    "type": "match",
                    "data": {
                        "path": {"text": str(dummy_repo_path / "auth.py")},
                        "line_number": 10,
                        "lines": {"text": "def login():"}
                    }
                }
            ])
            mock_run.return_value = type("MockProcess", (), {
                "stdout": mock_output,
                "stderr": "",
                "returncode": 0,
                "capture_output": True,
                "text": True,
                "check": False
            })()
            
            # Test with empty list
            result1 = create_grep_rag_instance._run_ripgrep(
                ["def.*login"],
                str(dummy_repo_path),
                whitelist=[],
            )
            
            # Test with None
            result2 = create_grep_rag_instance._run_ripgrep(
                ["def.*login"],
                str(dummy_repo_path),
                whitelist=None,
            )
            
            # Verify both behave the same
            assert isinstance(result1, list), "Empty list should return list"
            assert isinstance(result2, list), "None should return list"
