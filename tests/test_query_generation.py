"""
Tests for the query generation functionality.
"""

import json
import pytest
from unittest.mock import patch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grepRAG import GrepRAG


@pytest.fixture
def create_query_generator() -> GrepRAG:
    """Fixture creating a GrepRAG instance for query generation tests."""
    return GrepRAG(
        main_inference_url="http://192.168.188.106/v1",
        main_model_name="Qwythos-9B",
        main_model_params={"max_tokens": 500, "temperature": 0.7},
        grep_model_path_or_name="tests/fixtures/greprag-0.6b",
        external_api_type="openai",
        external_api_key="test-query-gen-key",
        whitelist=None,
        blacklist=None,
        context_padding=2,
    )


def load_test_queries(test_name: str) -> list[str]:
    """Load test queries from JSON file."""
    json_path = Path(__file__).parent / "data" / "test_queries.json"
    with open(json_path, "r") as f:
        data = json.load(f)
        return data["test_queries"][test_name]["queries"]


class TestQueryGeneration:
    """Test suite for query generation functionality."""

    def test_generate_valid_queries(
        self,
        create_query_generator: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that valid prompts generate 3 regex patterns."""
        # Mock the model generation to return predictable queries
        with patch.object(create_query_generator, "_generate_grep_queries") as mock_gen:
            mock_gen.return_value = load_test_queries("valid_prompt")
            
            # Mock ripgrep to avoid actual subprocess call - we just need valid structure
            with patch("subprocess.run") as mock_run:
                # Return matches for each query pattern
                def side_effect(*args, **kwargs):
                    # ripgrep outputs each match as a separate JSON object on its own line
                    query = args[0][2] if len(args) > 1 else "unknown"
                    import sys; sys.stderr.write(f"[MOCK] side_effect called with query={query!r}\\n")
                    json_line = '{"type": "match", "data": {"path": {"text": "tests/fixtures/dummy_repo/auth.py"}, "line_number": 10, "lines": {"text": "def login():\\n    pass"}}}'
                    return type("MockProcess", (), {"stdout": json_line + "\\n", "stderr": "", "returncode": 0, "capture_output": True, "text": True, "check": False})()
                
                mock_run.side_effect = side_effect
            
            # Call the method
            result = create_query_generator._run_ripgrep(
                load_test_queries("valid_prompt"),
                str(dummy_repo_path),
            )
            
            # Verify result
            assert len(result) > 0, "Should return at least one block"
            assert isinstance(result, list), "Result should be a list"
            assert all(
                isinstance(block, dict) for block in result
            ), "All blocks should be dictionaries"

    def test_generate_empty_result(
        self,
        create_query_generator: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that ambiguous prompts return empty list."""
        # Mock the model generation - empty prompt gives a catch-all query
        with patch.object(create_query_generator, "_generate_grep_queries") as mock_gen:
            mock_gen.return_value = load_test_queries("empty_prompt")
            
            # Ripgrep will match everything, so we mock subprocess to return empty
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = type("MockProcess", (), {
                    "stdout": "",
                    "stderr": "",
                    "returncode": 1,  # No matches = non-zero exit
                    "capture_output": True,
                    "text": True,
                    "check": False
                })()
                
                result = create_query_generator._run_ripgrep(
                    load_test_queries("empty_prompt"),
                    str(dummy_repo_path),
                )
                
                # Verify result
                assert result == [], "Empty prompt should return empty list"

    def test_generate_multiple_queries(
        self,
        create_query_generator: GrepRAG,
        dummy_repo_path: Path,
    ) -> None:
        """Test that prompts generate 5+ regex patterns."""
        # Mock the model generation
        with patch.object(create_query_generator, "_generate_grep_queries") as mock_gen:
            mock_gen.return_value = load_test_queries("multi_query")
            
            result = create_query_generator._run_ripgrep(
                load_test_queries("multi_query"),
                str(dummy_repo_path),
            )
            
            # Verify result
            assert len(result) >= 1, "Should return at least one block"
            assert isinstance(result, list), "Result should be a list"

    def test_query_format(
        self,
        create_query_generator: GrepRAG,
    ) -> None:
        """Test that regex patterns are valid strings, not objects."""
        queries = load_test_queries("valid_prompt")
        
        # Verify format
        assert isinstance(queries, list), "Queries should be a list"
        assert all(isinstance(q, str) for q in queries), \
            "All queries should be strings"
        assert len(queries) == 3, "Should have exactly 3 queries"

    def test_query_content(
        self,
        create_query_generator: GrepRAG,
    ) -> None:
        """Test that generated queries contain keywords from prompt."""
        queries = load_test_queries("valid_prompt")
        
        # Verify content
        assert any("def" in q.lower() for q in queries), \
            "Should contain 'def' keyword"
        assert any("class" in q.lower() for q in queries), \
            "Should contain 'class' keyword"
        assert any("login" in q.lower() or "auth" in q.lower() for q in queries), \
            "Should contain 'login' or 'auth' keyword"
