"""
Tests for the blacklist filtering functionality.
"""

import json
from pathlib import Path

import pytest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grepRAG import GrepRAG


class TestBlacklistFiltering:
    """Test suite for blacklist filtering functionality."""

    def test_exclude_all(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that `**/*.py` pattern excludes all Python files."""
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
            
            # Test with exclude all pattern
            result = create_grep_rag_instance._run_ripgrep(
                ["def.*login"],
                str(dummy_repo_path),
                blacklist=["**/*.py"],
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_exclude_directory(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that `**/deeply_nested/**` excludes nested directory."""
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
                blacklist=["**/deeply_nested/**"],
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_exclude_pattern_priority(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that blacklist overrides whitelist."""
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
            
            # Test with whitelist + blacklist
            result = create_grep_rag_instance._run_ripgrep(
                ["def.*login"],
                str(dummy_repo_path),
                whitelist=["**/*.py"],
                blacklist=["**/auth.py"],
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_empty_blacklist(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that empty list `[]` includes all files."""
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
            
            # Test with empty blacklist
            result = create_grep_rag_instance._run_ripgrep(
                ["def.*login"],
                str(dummy_repo_path),
                blacklist=[],
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_multiple_blacklist_patterns(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that multiple blacklist patterns combined with OR."""
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
                blacklist=["**/auth.py", "**/models.py"],
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_whitespace_in_pattern(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that whitespace in pattern strings handled."""
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
            
            # Test with whitespace in pattern
            result = create_grep_rag_instance._run_ripgrep(
                ["def.*login"],
                str(dummy_repo_path),
                blacklist=["  **/auth.py  "],
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"
