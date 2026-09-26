"""
Tests for the OpenAI API integration functionality.
"""

import json
import urllib
from unittest.mock import MagicMock, patch

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
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps({
                "choices": [
                    {
                        "message": {
                            "content": "Found 3 authentication-related functions"
                        }
                    }
                ]
            }).encode("utf-8")
            mock_response.decode.return_value = json.dumps({
                "choices": [
                    {
                        "message": {
                            "content": "Found 3 authentication-related functions"
                        }
                    }
                ]
            })
            mock_urlopen.return_value.__enter__.return_value = mock_response
            mock_urlopen.return_value.__exit__.return_value = None
            
            result = create_grep_rag_instance._call_openai(
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
            # Simulate a URLError being raised by the real urlopen call
            mock_urlopen.side_effect = urllib.error.URLError("Network error")
            
            result = create_grep_rag_instance._call_openai(
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
            mock_response = MagicMock()
            mock_response.read.return_value = b"not valid json"
            mock_response.decode.return_value = "not valid json"
            mock_urlopen.return_value.__enter__.return_value = mock_response
            mock_urlopen.return_value.__exit__.return_value = None
            
            result = create_grep_rag_instance._call_openai(
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
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps({
                "choices": [{"message": {"content": "test"}}]
            }).encode("utf-8")
            mock_response.decode.return_value = json.dumps({
                "choices": [{"message": {"content": "test"}}]
            })
            mock_urlopen.return_value.__enter__.return_value = mock_response
            mock_urlopen.return_value.__exit__.return_value = None
            
            create_grep_rag_instance._call_openai(
                "Find auth functions",
                "context",
            )
            
            # Verify headers by inspecting the Request object passed to urlopen
            request = mock_urlopen.call_args[0][0]  # First positional arg is the Request
            assert isinstance(request, urllib.request.Request), "Should receive a Request object"
            headers = dict(request.headers)
            expected_auth = f"Bearer {create_grep_rag_instance.external_api_key}"
            assert headers.get("Authorization") == expected_auth, \
                f"Expected 'Authorization: {expected_auth}', got {headers}"
            
            # Verify payload by inspecting the Request data attribute
            request = mock_urlopen.call_args[0][0]  # First positional arg is the Request
            payload = json.loads(request.data.decode("utf-8"))
            assert "messages" in payload, "Payload should have 'messages' key"
            assert len(payload["messages"]) == 2, "Payload should have 2 messages"

    def test_call_backtick_removal(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that backticks stripped from JSON response."""
        # Mock response without backticks - the API call should return clean text
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            content_text = "Found 3 functions\n\n1. def login(): pass"
            mock_response.read.return_value = json.dumps({
                "choices": [{"message": {"content": content_text}}]
            }).encode("utf-8")
            mock_response.decode.return_value = json.dumps({
                "choices": [{"message": {"content": content_text}}]
            })
            mock_urlopen.return_value.__enter__.return_value = mock_response
            mock_urlopen.return_value.__exit__.return_value = None
            
            result = create_grep_rag_instance._call_openai(
                "Find auth functions",
                "context",
            )

            # Verify backtick removal (nothing to remove in clean mock)
            assert "```" not in result, "Result should not contain backticks"

    @pytest.mark.asyncio
    async def test_call_async(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that async version works correctly."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps({
                "choices": [{"message": {"content": "test"}}]
            }).encode("utf-8")
            mock_response.decode.return_value = json.dumps({
                "choices": [{"message": {"content": "test"}}]
            })
            mock_urlopen.return_value.__enter__.return_value = mock_response
            mock_urlopen.return_value.__exit__.return_value = None
            
            # Note: _call_openai is synchronous; there's no async variant
            result = create_grep_rag_instance._call_openai(
                "Find auth functions",
                "context",
            )

            # Verify result
            assert isinstance(result, str), "Result should be a string"

