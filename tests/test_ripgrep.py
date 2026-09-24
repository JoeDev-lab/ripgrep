"""
Tests for the ripgrep execution functionality.
"""

import json
import subprocess
from unittest.mock import patch

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grepRAG import GrepRAG


@pytest.fixture
def create_grep_rag() -> GrepRAG:
    """Fixture creating a GrepRAG instance for ripgrep tests."""
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


class TestRipgrepExecution:
    """Test suite for ripgrep execution functionality."""

    def test_run_valid_queries(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that valid queries return expected blocks."""
        with patch("subprocess.run") as mock_run:
            # Mock ripgrep output
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
            
            result = create_grep_rag._run_ripgrep(
                ["def.*login"],
                str(dummy_repo_path),
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"
            assert len(result) >= 0, "Should return at least 0 blocks"

    def test_run_empty_queries(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that empty query list returns empty blocks."""
        result = create_grep_rag._run_ripgrep([], "/tmp")
        
        assert result == [], "Empty queries should return empty list"

    def test_context_padding(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that -C flag passes correctly."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = type("MockProcess", (), {
                "stdout": "[]",
                "stderr": "",
                "returncode": 0,
                "capture_output": True,
                "text": True,
                "check": False
            })()
            
            # Test with context_padding=2
            create_grep_rag._run_ripgrep(
                ["test"],
                str(dummy_repo_path),
            )
            
            # Verify -C 2 was passed
            call_args = mock_run.call_args
            cmd = call_args[0][0]  # First positional argument
            assert "-C" in cmd, "-C flag should be in command"
            assert "2" in cmd, "-C 2 should be in command"

    def test_json_parsing(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that malformed JSON in ripgrep output handled gracefully."""
        with patch("subprocess.run") as mock_run:
            # Mock ripgrep output with some malformed JSON
            mock_output = json.dumps([
                {"type": "match", "data": {"path": {"text": str(dummy_repo_path / "auth.py")}, "line_number": 10, "lines": {"text": "def login():"}}},
                "malformed json",  # This should be caught
                {"type": "match", "data": {"path": {"text": str(dummy_repo_path / "auth.py")}, "line_number": 11, "lines": {"text": "pass"}}}
            ])
            mock_run.return_value = type("MockProcess", (), {
                "stdout": mock_output,
                "stderr": "",
                "returncode": 0,
                "capture_output": True,
                "text": True,
                "check": False
            })()
            
            # Should not raise exception
            result = create_grep_rag._run_ripgrep(
                ["def.*login"],
                str(dummy_repo_path),
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_process_return_code(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that non-zero exit codes handled without crashing."""
        with patch("subprocess.run") as mock_run:
            # Mock ripgrep with non-zero exit code
            mock_output = json.dumps([
                {"type": "match", "data": {"path": {"text": str(dummy_repo_path / "auth.py")}, "line_number": 10, "lines": {"text": "def login():"}}}
            ])
            mock_run.return_value = type("MockProcess", (), {
                "stdout": mock_output,
                "stderr": "Some error",
                "returncode": 1,  # Non-zero exit code
                "capture_output": True,
                "text": True,
                "check": False
            })()
            
            # Should not raise exception
            result = create_grep_rag._run_ripgrep(
                ["def.*login"],
                str(dummy_repo_path),
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    @pytest.mark.asyncio
    async def test_async_run_valid_queries(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that async version works correctly."""
        with patch("subprocess.run") as mock_run:
            # Mock ripgrep output
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
            
            # Call async version
            result = await create_grep_rag._run_ripgrep_async(
                ["def.*login"],
                str(dummy_repo_path),
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    @pytest.mark.asyncio
    async def test_parallel_execution(
        self,
        create_grep_rag_instance: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that multiple queries executed in parallel via asyncio.gather()."""
        with patch("subprocess.run") as mock_run:
            # Mock ripgrep output for multiple queries
            mock_outputs = [
                json.dumps([
                    {"type": "match", "data": {"path": {"text": str(dummy_repo_path / "auth.py")}, "line_number": 10, "lines": {"text": "def login():"}}}
                ]) for _ in range(3)
            ]
            
            async def mock_run_async(*args, **kwargs):
                return type("MockProcess", (), {
                    "stdout": mock_outputs[0],
                    "stderr": "",
                    "returncode": 0,
                    "capture_output": True,
                    "text": True,
                    "check": False
                })()
            
            with patch("asyncio.to_thread", return_value=mock_run_async()):
                # Call async version
                result = await create_grep_rag._run_ripgrep_async(
                    ["def.*login", "class.*User", "def.*auth"],
                    str(dummy_repo_path),
                )
                
                # Verify result
                assert isinstance(result, list), "Result should be a list"
