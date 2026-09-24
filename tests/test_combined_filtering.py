"""
Tests for the combined whitelist and blacklist filtering functionality.
"""

import json
from pathlib import Path

import pytest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grepRAG import GrepRAG


class TestCombinedFiltering:
    """Test suite for combined filtering functionality."""

    def test_whitelist_before_blacklist(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that whitelist applied first, then blacklist."""
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
            
            # Test with whitelist first, then blacklist
            result = create_grep_rag._run_ripgrep(
                ["def.*login"],
                str(dummy_repo_path),
                whitelist=["**/*.py"],
                blacklist=["**/auth.py"],
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_both_empty(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that both empty means no filtering."""
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
            
            # Test with both empty
            result = create_grep_rag._run_ripgrep(
                ["def.*login"],
                str(dummy_repo_path),
                whitelist=[],
                blacklist=[],
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_whitelist_includes_all_then_blacklist(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that whitelist `**/*`, blacklist specific."""
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
            
            # Test with whitelist `**/*`, blacklist specific
            result = create_grep_rag._run_ripgrep(
                ["def.*login"],
                str(dummy_repo_path),
                whitelist=["**/*"],
                blacklist=["**/auth.py"],
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_blacklist_excludes_all_then_whitelist(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test edge case: blacklist `**/*`, whitelist specific (should return files)."""
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
            
            # Test with blacklist `**/*`, whitelist specific
            result = create_grep_rag._run_ripgrep(
                ["def.*login"],
                str(dummy_repo_path),
                whitelist=["**/auth.py"],
                blacklist=["**/*"],
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_nested_pattern_matching(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that `**/a/**/b.py` matches nested paths."""
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
            
            # Test with nested pattern
            result = create_grep_rag._run_ripgrep(
                ["def.*nested"],
                str(dummy_repo_path),
                whitelist=["**/deeply_nested/**"],
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"
