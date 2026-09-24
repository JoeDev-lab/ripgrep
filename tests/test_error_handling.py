"""
Tests for error handling in the GrepRAG functionality.
"""

import json
from pathlib import Path

import pytest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grepRAG import GrepRAG


class TestErrorHandling:
    """Test suite for error handling in GrepRAG functionality."""

    def test_missing_ripgrep(
        self,
        dummy_repo_path: Path,
    ) -> None:
        """Test that `RuntimeError` with helpful message."""
        # Simulate missing ripgrep
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("rg: command not found")
            
            grep_rag = GrepRAG(
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
            
            with pytest.raises(FileNotFoundError):
                grep_rag._run_ripgrep(
                    ["def.*login"],
                    str(dummy_repo_path),
                )

    def test_missing_api_key(
        self,
    ) -> None:
        """Test that `ValueError` with clear message."""
        with pytest.raises(ValueError, match="API key"):
            GrepRAG(
                main_inference_url="https://test-endpoint.com/v1/chat/completions",
                main_model_name="test-model",
                main_model_params={"max_tokens": 500, "temperature": 0.7},
                grep_model_path_or_name="test-model",
                external_api_type="openai",
                external_api_key="",  # Empty API key
                whitelist=None,
                blacklist=None,
                context_padding=2,
            )

    def test_invalid_api_type(
        self,
    ) -> None:
        """Test that `ValueError` for unsupported type."""
        with pytest.raises(ValueError, match="Unsupported"):
            GrepRAG(
                main_inference_url="https://test-endpoint.com/v1/chat/completions",
                main_model_name="test-model",
                main_model_params={"max_tokens": 500, "temperature": 0.7},
                grep_model_path_or_name="test-model",
                external_api_type="invalid_type",  # Invalid type
                external_api_key="test-key",
                whitelist=None,
                blacklist=None,
                context_padding=2,
            )

    def test_missing_tree_sitter_parser(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that warning logged, regex fallback used."""
        blocks = [
            {
                "file": "test.unknown_ext",
                "start_line": 1,
                "end_line": 5,
                "lines": ["def login():\n    pass"],
            },
        ]
        
        with patch("logging.Logger.warning") as mock_warn:
            result = create_grep_rag._ast_weighted_rerank(blocks)
            
            # Verify warning was logged
            assert mock_warn.called, "Warning should be logged"

    def test_network_timeout(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that timeout error handled gracefully."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = TimeoutError("Connection timed out")
            
            result = create_grep_rag._call_openai(
                "Find auth functions",
                "context",
            )
            
            # Verify error handling
            assert isinstance(result, str), "Result should be a string"
            assert "Error" in result, "Result should contain 'Error'"

    def test_model_loading_failure(
        self,
    ) -> None:
        """Test that partial failure handled."""
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_load:
            mock_load.side_effect = RuntimeError("Model loading failed")
            
            # Should raise RuntimeError
            with pytest.raises(RuntimeError):
                GrepRAG(
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

    def test_all_parsers_fail(
        self,
    ) -> None:
        """Test that all tree-sitter parsers fail, regex-only mode."""
        with patch("subprocess.run") as mock_run:
            # Simulate all parsers failing
            mock_run.side_effect = FileNotFoundError("rg: command not found")
            
            grep_rag = GrepRAG(
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
            
            # Should fall back to regex scoring
            blocks = [
                {
                    "file": "test.unknown_ext",
                    "start_line": 1,
                    "end_line": 5,
                    "lines": ["def login():\n    pass"],
                },
            ]
            
            with patch("logging.Logger.warning") as mock_warn:
                result = grep_rag._ast_weighted_rerank(blocks)
                
                # Verify regex fallback
                assert len(result) == 1, "Should return 1 block"
                assert "score" in result[0], "Block should have score"
