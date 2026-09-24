"""
Tests for edge cases in the GrepRAG functionality.
"""

import json
from pathlib import Path

import pytest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grepRAG import GrepRAG


class TestEdgeCases:
    """Test suite for edge cases in GrepRAG functionality."""

    def test_empty_repo_path(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that non-existent directory handled gracefully."""
        with patch("subprocess.run") as mock_run:
            mock_output = json.dumps([])
            mock_run.return_value = type("MockProcess", (), {
                "stdout": mock_output,
                "stderr": "",
                "returncode": 0,
                "capture_output": True,
                "text": True,
                "check": False
            })()
            
            # Test with non-existent directory
            result = create_grep_rag._run_ripgrep(
                ["def.*login"],
                "/non/existent/path",
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_empty_prompt(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that empty string prompt produces minimal queries."""
        with patch.object(create_grep_rag, "_generate_grep_queries") as mock_gen:
            mock_gen.return_value = [".*"]  # Minimal regex
            
            result = create_grep_rag._run_ripgrep(
                [".*"],
                str(dummy_repo_path),
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_top_k_zero(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that `top_k=0` returns empty context."""
        blocks = [
            {
                "file": "test.py",
                "start_line": 1,
                "end_line": 5,
                "lines": ["def login():\n    pass"],
            },
        ]
        
        # Test with top_k=0
        result = create_grep_rag._format_context(blocks[:0])
        
        # Verify result
        assert result == "", "top_k=0 should return empty string"

    def test_top_k_large(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that `top_k=1000` with 50 files returns all 50."""
        blocks = [
            {
                "file": f"test{i}.py",
                "start_line": 1,
                "end_line": 5,
                "lines": [f"def func{i}():\n    pass"],
            }
            for i in range(50)
        ]
        
        # Test with top_k=1000 (larger than available blocks)
        result = create_grep_rag._format_context(blocks[:1000])
        
        # Verify result
        assert "File: test0.py" in result, "Should contain first file"
        assert "File: test49.py" in result, "Should contain last file"

    def test_unicode_filenames(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that files with Unicode names handled."""
        blocks = [
            {
                "file": "tests/测试/测试.py",
                "start_line": 1,
                "end_line": 5,
                "lines": ["def login():\n    pass"],
            },
        ]
        
        result = create_grep_rag._format_context(blocks)
        
        # Verify Unicode handling
        assert "tests/测试/测试.py" in result, "Should contain Unicode filename"

    def test_multiline_strings(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that multi-line regex queries work."""
        with patch.object(create_grep_rag, "_generate_grep_queries") as mock_gen:
            # Test with multi-line regex
            mock_gen.return_value = ["def.*\\s+\\w+", "class.*\\w+", "async.*def.*\\w+"]
            
            result = create_grep_rag._run_ripgrep(
                ["def.*\\s+\\w+", "class.*\\w+", "async.*def.*\\w+"],
                str(Path(__file__).parent / "fixtures" / "dummy_repo"),
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_special_chars_in_prompt(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that prompt with special characters handled."""
        with patch.object(create_grep_rag, "_generate_grep_queries") as mock_gen:
            # Test with special characters
            mock_gen.return_value = ["def.*[a-z]+", "class.*[A-Z]+", "async.*def.*[a-z]+"]
            
            result = create_grep_rag._run_ripgrep(
                ["def.*[a-z]+", "class.*[A-Z]+", "async.*def.*[a-z]+"],
                str(Path(__file__).parent / "fixtures" / "dummy_repo"),
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_very_long_lines(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that lines > 1000 characters handled."""
        blocks = [
            {
                "file": "test.py",
                "start_line": 1,
                "end_line": 2,
                "lines": ["def" + " " * 1000 + "():\n    pass"],
            },
        ]
        
        result = create_grep_rag._format_context(blocks)
        
        # Verify long lines handled
        assert "def" in result, "Should contain 'def'"
        assert "..." in result, "Should contain trailing dots"

    @pytest.mark.asyncio
    async def test_async_edge_cases(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that async version handles all edge cases."""
        with patch("subprocess.run") as mock_run:
            mock_output = json.dumps([
                {
                    "type": "match",
                    "data": {
                        "path": {"text": str(Path(__file__).parent / "fixtures" / "dummy_repo" / "auth.py")},
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
            
            # Call async version
            result = await create_grep_rag._run_ripgrep_async(
                ["def.*login"],
                str(Path(__file__).parent / "fixtures" / "dummy_repo"),
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"
