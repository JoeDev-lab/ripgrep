"""
Tests for the OpenAI API integration functionality.
"""

import json
from unittest.mock import patch

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grepRAG import GrepRAG


class TestOpenAIIntegration:
    """Test suite for OpenAI API integration functionality."""

    def test_call_valid_response(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that valid response returns expected message structure."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = type("MockResponse", (), {
                "read": lambda self: json.dumps({
                    "choices": [
                        {
                            "message": {
                                "content": "Found 3 authentication-related functions"
                            }
                        }
                    ]
                }).encode("utf-8"),
                "decode": lambda self: json.dumps({
                    "choices": [
                        {
                            "message": {
                                "content": "Found 3 authentication-related functions"
                            }
                        }
                    ]
                }),
            })()
            mock_urlopen.return_value = mock_response
            
            result = create_grep_rag._call_openai(
                "Find auth functions",
                "context",
            )
            
            # Verify result
            assert isinstance(result, str), "Result should be a string"
            assert "Found" in result, "Result should contain 'Found'"

    def test_call_error_handling(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that network error returns error string."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = Exception("Network error")
            
            result = create_grep_rag._call_openai(
                "Find auth functions",
                "context",
            )
            
            # Verify error handling
            assert isinstance(result, str), "Result should be a string"
            assert "Error" in result, "Result should contain 'Error'"

    def test_call_json_parsing(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that malformed JSON response handled."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = type("MockResponse", (), {
                "read": lambda self: "not valid json".encode("utf-8"),
                "decode": lambda self: "not valid json",
            })()
            mock_urlopen.return_value = mock_response
            
            result = create_grep_rag._call_openai(
                "Find auth functions",
                "context",
            )
            
            # Verify error handling
            assert isinstance(result, str), "Result should be a string"

    def test_call_headers(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that correct Authorization header format."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = type("MockResponse", (), {
                "read": lambda self: json.dumps({
                    "choices": [{"message": {"content": "test"}}]
                }).encode("utf-8"),
                "decode": lambda self: json.dumps({
                    "choices": [{"message": {"content": "test"}}]
                }),
            })()
            mock_urlopen.return_value = mock_response
            
            create_grep_rag._call_openai(
                "Find auth functions",
                "context",
            )
            
            # Verify headers
            call_args = mock_urlopen.call_args
            headers = call_args[1]["headers"]
            assert headers.get("Authorization") == "Bearer test-key"

    def test_call_payload_structure(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that messages array includes system + user."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = type("MockResponse", (), {
                "read": lambda self: json.dumps({
                    "choices": [{"message": {"content": "test"}}]
                }).encode("utf-8"),
                "decode": lambda self: json.dumps({
                    "choices": [{"message": {"content": "test"}}]
                }),
            })()
            mock_urlopen.return_value = mock_response
            
            create_grep_rag._call_openai(
                "Find auth functions",
                "context",
            )
            
            # Verify payload
            call_args = mock_urlopen.call_args
            payload = json.loads(call_args[1]["data"])
            assert "messages" in payload, "Payload should have 'messages' key"
            assert len(payload["messages"]) == 2, "Payload should have 2 messages"

    def test_call_backtick_removal(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that backticks stripped from JSON response."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = type("MockResponse", (), {
                "read": lambda self: json.dumps({
                    "choices": [
                        {
                            "message": {
                                "content": "Found 3 functions\n\n1. ```def login()```"
                            }
                        }
                    ]
                }).encode("utf-8"),
                "decode": lambda self: json.dumps({
                    "choices": [
                        {
                            "message": {
                                "content": "Found 3 functions\n\n1. ```def login()```"
                            }
                        }
                    ]
                }),
            })()
            mock_urlopen.return_value = mock_response
            
            result = create_grep_rag._call_openai(
                "Find auth functions",
                "context",
            )
            
            # Verify backtick removal
            assert "```" not in result, "Result should not contain backticks"

    @pytest.mark.asyncio
    async def test_call_async(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that async version works correctly."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = type("MockResponse", (), {
                "read": lambda self: json.dumps({
                    "choices": [{"message": {"content": "test"}}]
                }).encode("utf-8"),
                "decode": lambda self: json.dumps({
                    "choices": [{"message": {"content": "test"}}]
                }),
            })()
            mock_urlopen.return_value = mock_response
            
            result = await create_grep_rag._call_openai(
                "Find auth functions",
                "context",
            )
            
            # Verify result
            assert isinstance(result, str), "Result should be a string"
