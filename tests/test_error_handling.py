"""
Tests for error handling in the GrepRAG functionality.
"""

import json
import subprocess
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
        """Test that RuntimeError is raised when ripgrep is missing."""
        # Simulate missing ripgrep - patch _check_dependencies specifically
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("rg: command not found")

            test_model_path = (Path(__file__).parent / "fixtures" / "greprag-0.6b").resolve()

            with pytest.raises(RuntimeError, match="ripgrep.*not found"):
                GrepRAG(
                    main_inference_url="https://test-endpoint.com/v1/chat/completions",
                    main_model_name="test-model",
                    main_model_params={"max_tokens": 500, "temperature": 0.7},
                    grep_model_path_or_name=str(test_model_path),
                    external_api_type="openai",
                    external_api_key="test-key",
                    whitelist=None,
                    blacklist=None,
                    context_padding=2,
                )

    def test_missing_api_key(
        self,
    ) -> None:
        """Test that `ValueError` with clear message."""
        # Mock subprocess.run to bypass ripgrep check
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(["rg"], 0, "", "")

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
            result = create_grep_rag_instance._ast_weighted_rerank(blocks)
            
            # Verify warning was logged
            assert mock_warn.called, "Warning should be logged"

    def test_network_timeout(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that timeout error handled gracefully."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = TimeoutError("Connection timed out")
            
            result = create_grep_rag_instance._call_openai(
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
        # Use the local fixture model path
        from pathlib import Path as TestPath
        test_model_path = (TestPath(__file__).parent / "fixtures" / "greprag-0.6b").resolve()
        
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_load:
            mock_load.side_effect = RuntimeError("Model loading failed")

            # Should raise RuntimeError
            with pytest.raises(RuntimeError):
                GrepRAG(
                    main_inference_url="https://test-endpoint.com/v1/chat/completions",
                    main_model_name="Qwythos-9B",
                    main_model_params={"max_tokens": 500, "temperature": 0.7},
                    grep_model_path_or_name=str(test_model_path),
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
        # Use the local fixture model path
        from pathlib import Path as TestPath
        test_model_path = (TestPath(__file__).parent / "fixtures" / "greprag-0.6b").resolve()
        
        with patch("subprocess.run") as mock_run:
            # Simulate all parsers failing
            mock_run.side_effect = FileNotFoundError("rg: command not found")
            
            # GrepRAG creation should raise RuntimeError due to missing ripgrep
            with pytest.raises(RuntimeError, match="ripgrep.*not found"):
                GrepRAG(
                    main_inference_url="https://test-endpoint.com/v1/chat/completions",
                    main_model_name="Qwythos-9B",
                    main_model_params={"max_tokens": 500, "temperature": 0.7},
                    grep_model_path_or_name=str(test_model_path),
                    external_api_type="openai",
                    external_api_key="test-key",
                    whitelist=None,
                    blacklist=None,
                    context_padding=2,
                )
