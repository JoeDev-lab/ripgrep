"""
Tests for the context formatting functionality.
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grepRAG import GrepRAG


class TestContextFormatting:
    """Test suite for context formatting functionality."""

    def test_format_single_block(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that single block formatted correctly."""
        blocks = [
            {
                "file": "test.py",
                "start_line": 1,
                "end_line": 5,
                "lines": ["def login():\n    pass"],
            },
        ]
        
        result = create_grep_rag_instance._format_context(blocks)
        
        # Verify formatting
        assert "File: test.py" in result, "Should contain file name"
        assert "Lines 1-5" in result, "Should contain line range"
        assert "def login():" in result, "Should contain code content"

    def test_format_multiple_blocks(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that multiple blocks separated by delimiters."""
        blocks = [
            {
                "file": "test1.py",
                "start_line": 1,
                "end_line": 5,
                "lines": ["def login():\n    pass"],
            },
            {
                "file": "test2.py",
                "start_line": 10,
                "end_line": 15,
                "lines": ["def logout():\n    pass"],
            },
        ]
        
        result = create_grep_rag_instance._format_context(blocks)
        
        # Verify formatting
        assert "File: test1.py" in result, "Should contain first file"
        assert "File: test2.py" in result, "Should contain second file"
        assert "Lines 1-5" in result, "Should contain first line range"
        assert "Lines 10-15" in result, "Should contain second line range"

    def test_format_line_ranges(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that line numbers included in output."""
        blocks = [
            {
                "file": "test.py",
                "start_line": 1,
                "end_line": 5,
                "lines": ["def login():\n    pass"],
            },
        ]
        
        result = create_grep_rag_instance._format_context(blocks)
        
        # Verify line ranges
        assert "Lines 1-5" in result, "Should contain line range"

    def test_format_empty_blocks(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that empty line arrays handled."""
        blocks = [
            {
                "file": "test.py",
                "start_line": 1,
                "end_line": 5,
                "lines": [],
            },
        ]
        
        result = create_grep_rag_instance._format_context(blocks)
        
        # Verify empty blocks handled
        assert "File: test.py" in result, "Should contain file name"

    def test_format_trailing_dots(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that '...' suffix present after each block."""
        blocks = [
            {
                "file": "test.py",
                "start_line": 1,
                "end_line": 5,
                "lines": ["def login():\n    pass"],
            },
        ]
        
        result = create_grep_rag_instance._format_context(blocks)
        
        # Verify trailing dots
        assert "..." in result, "Should contain trailing dots"
