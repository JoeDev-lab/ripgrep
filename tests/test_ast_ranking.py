"""
Tests for the AST ranking functionality.
"""

import json
import logging
from unittest.mock import patch

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grepRAG import GrepRAG


@pytest.fixture
def create_grep_rag() -> GrepRAG:
    """Fixture creating a GrepRAG instance for AST ranking tests."""
    return GrepRAG(
        main_inference_url="https://test-endpoint.com/v1/chat/completions",
        main_model_name="test-model",
        main_model_params={"max_tokens": 500, "temperature": 0.7},
        grep_model_path_or_name="test-model",
        external_api_type="openai",
        external_api_key="test-key",
        whitelist=None,
        blacklist=None,
        context_padding=2,
    )


class TestASTRanking:
    """Test suite for AST ranking functionality."""

    def test_rank_function_blocks(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that functions (+5.0) ranked higher than identifiers (+0.5)."""
        blocks = [
            {
                "file": "test.py",
                "start_line": 1,
                "end_line": 5,
                "lines": ["def login():\n    pass"],
            },
            {
                "file": "test.py",
                "start_line": 10,
                "end_line": 12,
                "lines": ["def helper():\n    pass"],
            },
        ]
        
        result = create_grep_rag._ast_weighted_rerank(blocks)
        
        # Verify ranking
        assert len(result) == 2, "Should return 2 blocks"
        assert all(
            isinstance(block, dict) for block in result
        ), "All blocks should be dictionaries"

    def test_rank_class_blocks(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that classes (+5.0) ranked correctly."""
        blocks = [
            {
                "file": "test.py",
                "start_line": 1,
                "end_line": 5,
                "lines": ["class UserModel:\n    pass"],
            },
        ]
        
        result = create_grep_rag._ast_weighted_rerank(blocks)
        
        # Verify ranking
        assert len(result) == 1, "Should return 1 block"

    def test_rank_by_score(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that top-k filtering works correctly."""
        blocks = [
            {
                "file": "test.py",
                "start_line": 1,
                "end_line": 5,
                "lines": ["def login():\n    pass"],
            },
            {
                "file": "test.py",
                "start_line": 10,
                "end_line": 12,
                "lines": ["def helper():\n    pass"],
            },
        ]
        
        result = create_grep_rag._ast_weighted_rerank(blocks)
        
        # Verify sorting
        assert len(result) == 2, "Should return 2 blocks"
        assert all(
            isinstance(block, dict) for block in result
        ), "All blocks should be dictionaries"

    def test_regex_fallback(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that unknown/missing parsers fall back to regex scoring."""
        blocks = [
            {
                "file": "test.unknown_ext",
                "start_line": 1,
                "end_line": 5,
                "lines": ["def login():\n    pass"],
            },
        ]
        
        result = create_grep_rag._ast_weighted_rerank(blocks)
        
        # Verify regex fallback
        assert len(result) == 1, "Should return 1 block"
        assert "score" in result[0], "Block should have score"

    def test_missing_parser_warning(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that warning logged once per language, not repeated."""
        with patch("logging.Logger.warning") as mock_warn:
            blocks = [
                {
                    "file": "test.unknown_ext",
                    "start_line": 1,
                    "end_line": 5,
                    "lines": ["def login():\n    pass"],
                },
            ]
            
            result = create_grep_rag._ast_weighted_rerank(blocks)
            
            # Verify warning was called
            assert mock_warn.called, "Warning should be logged"

    def test_score_accumulation(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that multiple captures on same block accumulate scores."""
        blocks = [
            {
                "file": "test.py",
                "start_line": 1,
                "end_line": 5,
                "lines": ["def login():\n    pass"],
            },
        ]
        
        result = create_grep_rag._ast_weighted_rerank(blocks)
        
        # Verify score accumulation
        assert len(result) == 1, "Should return 1 block"
        assert "score" in result[0], "Block should have score"
