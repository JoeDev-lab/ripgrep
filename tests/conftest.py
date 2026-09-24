"""
Base test utilities and fixtures for GrepRAG testing.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent / "src" / "greprag"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from grepRAG import GrepRAG


logger = logging.getLogger(__name__)


@pytest.fixture
def dummy_repo_path() -> Path:
    """
    Fixture returning the path to the dummy repository.
    
    Returns:
        Path: Path to the dummy repo directory.
    """
    return Path(__file__).parent / "fixtures" / "dummy_repo"


@pytest.fixture
def create_grep_rag_instance(dummy_repo_path: Path) -> GrepRAG:
    """
    Fixture that creates a GrepRAG instance with mocked dependencies.
    
    Returns:
        GrepRAG: A GrepRAG instance configured for testing.
    """
    # Use an absolute path to the test model
    # __file__ is in tests/conftest.py, so parent is tests
    test_model_path = (Path(__file__).parent / "fixtures" / "test_model" / "greprag-0.6b")
    # Resolve to get the absolute path
    test_model_path = test_model_path.resolve()
    
    return GrepRAG(
        main_inference_url="https://test-endpoint.com/v1/chat/completions",
        main_model_name="test-model",
        main_model_params={
            "max_tokens": 500,
            "temperature": 0.7,
        },
        grep_model_path_or_name=test_model_path,
        external_api_type="openai",
        external_api_key="test-key",
        whitelist=None,
        blacklist=None,
        context_padding=2,
    )


@pytest.fixture
def load_test_data(json_path: str) -> dict:
    """
    Fixture to load JSON test data.
    
    Args:
        json_path: Path to the JSON file.
        
    Returns:
        dict: Parsed JSON data.
    """
    with open(json_path, "r") as f:
        return json.load(f)


@pytest.fixture
def mock_api_response() -> str:
    """
    Fixture with a valid API response.
    
    Returns:
        str: Sample API response string.
    """
    return {
        "choices": [
            {
                "message": {
                    "content": "Found 3 authentication-related functions"
                }
            }
        ]
    }


@pytest.fixture
def mock_anthropic_response() -> list:
    """
    Fixture with a valid Anthropic API response.
    
    Returns:
        list: Sample Anthropic API response.
    """
    return [
        {
            "content": [
                {
                    "text": "Found 3 authentication-related functions"
                }
            ]
        }
    ]


def assert_block_structure(block: dict[str, Any], expected_file: str = "") -> None:
    """
    Assert that a block has the expected structure.
    
    Args:
        block: The block dictionary to validate.
        expected_file: Optional expected file path.
        
    Raises:
        AssertionError: If the block structure is invalid.
    """
    # Check required fields
    assert "file" in block, "Block must have 'file' field"
    assert "start_line" in block, "Block must have 'start_line' field"
    assert "end_line" in block, "Block must have 'end_line' field"
    assert "lines" in block, "Block must have 'lines' field"
    
    # Check types
    assert isinstance(block["file"], str), "File must be a string"
    assert isinstance(block["start_line"], int), "start_line must be an integer"
    assert isinstance(block["end_line"], int), "end_line must be an integer"
    assert isinstance(block["lines"], list), "lines must be a list"
    
    # Check optional fields
    if "score" in block:
        assert isinstance(block["score"], (int, float)), "score must be a number"
    
    # Check optional file path
    if expected_file:
        assert block["file"] == expected_file, f"Expected file '{expected_file}', got '{block['file']}'"


def assert_ranking_order(blocks: list[dict[str, Any]], expected_order: list[str]) -> None:
    """
    Assert that blocks are ranked in the expected order.
    
    Args:
        blocks: The list of blocks to validate.
        expected_order: List of file paths in expected order.
        
    Raises:
        AssertionError: If the ranking order is incorrect.
    """
    assert len(blocks) == len(expected_order), \
        f"Expected {len(expected_order)} blocks, got {len(blocks)}"
    
    for i, (block, expected) in enumerate(zip(blocks, expected_order)):
        assert block["file"] == expected, \
            f"Block {i} should be '{expected}', got '{block['file']}'"


def run_ripgrep_test(
    queries: list[str],
    repo_path: Path,
    expected_file_count: int = 1,
    context_padding: int = 2
) -> list[dict[str, Any]]:
    """
    Helper function to run a ripgrep test.
    
    Args:
        queries: List of regex queries.
        repo_path: Path to the repository.
        expected_file_count: Expected number of files found.
        context_padding: Context padding to use.
        
    Returns:
        list: List of blocks found.
    """
    cmd: list[str] = ["rg", "--json", "-C", str(context_padding)]
    
    for q in queries:
        cmd.extend(["-e", q])
    
    for wl in ["**/auth.py", "**/models.py"]:
        cmd.extend(["-g", wl])
    
    for bl in ["**/config.py"]:
        cmd.extend(["-g", f"!{bl}"])
    
    cmd.append(str(repo_path))
    
    blocks: list[dict[str, Any]] = []
    current_block: dict[str, Any] | None = None
    last_line_num: int = -1
    
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=False)
        for line in process.stdout.splitlines():
            if not line.strip():
                continue
            try:
                parsed: dict[str, Any] = json.loads(line)
                msg_type: str = parsed.get("type", "")
                
                if msg_type in ("match", "context"):
                    data: dict[str, Any] = parsed.get("data", {})
                    file_path: str = data.get("path", {}).get("text", "")
                    line_num: int = data.get("line_number", 0)
                    content: str = data.get("lines", {}).get("text", "").rstrip("\n")
                    
                    if current_block is None or current_block["file"] != file_path or line_num > last_line_num + 1:
                        if current_block is not None:
                            blocks.append(current_block)
                        current_block = {
                            "file": file_path,
                            "start_line": line_num,
                            "end_line": line_num,
                            "lines": [content]
                        }
                    else:
                        current_block["lines"].append(content)
                        current_block["end_line"] = line_num
                    
                    last_line_num = line_num
            except json.JSONDecodeError:
                continue
                
        if current_block is not None:
            blocks.append(current_block)
            
    except Exception:
        logger.error("Ripgrep execution failed: %s", sys.exc_info()[1])
        
    return blocks
