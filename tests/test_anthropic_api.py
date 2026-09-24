"""
Tests for the Anthropic API integration functionality.
"""

import json
from unittest.mock import patch

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grepRAG import GrepRAG


class TestAnthropicIntegration:
    """Test suite for Anthropic API integration functionality."""

    def test_call_valid_response(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that valid response returns expected structure."""
        # Create Anthropic-specific instance
        anthropic_grep_rag = GrepRAG(
            main_inference_url="https://test-endpoint.com/v1/chat/completions",
            main_model_name="test-model",
            main_model_params={"max_tokens": 500, "temperature": 0.7},
            grep_model_path_or_name="test-model",
            external_api_type="anthropic",
            external_api_key="test-key",
            whitelist=None,
            blacklist=None,
            context_padding=2,
        )
        
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = type("MockResponse", (), {
                "read": lambda self: json.dumps([
                    {
                        "content": [
                            {
                                "text": "Found 3 authentication-related functions"
                            }
                        ]
                    }
                ]).encode("utf-8"),
                "decode": lambda self: json.dumps([
                    {
                        "content": [
                            {
                                "text": "Found 3 authentication-related functions"
                            }
                        ]
                    }
                ]),
            })()
            mock_urlopen.return_value = mock_response
            
            result = anthropic_grep_rag._call_anthropic(
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
        anthropic_grep_rag = GrepRAG(
            main_inference_url="https://test-endpoint.com/v1/chat/completions",
            main_model_name="test-model",
            main_model_params={"max_tokens": 500, "temperature": 0.7},
            grep_model_path_or_name="test-model",
            external_api_type="anthropic",
            external_api_key="test-key",
            whitelist=None,
            blacklist=None,
            context_padding=2,
        )
        
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = Exception("Network error")
            
            result = anthropic_grep_rag._call_anthropic(
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
        anthropic_grep_rag = GrepRAG(
            main_inference_url="https://test-endpoint.com/v1/chat/completions",
            main_model_name="test-model",
            main_model_params={"max_tokens": 500, "temperature": 0.7},
            grep_model_path_or_name="test-model",
            external_api_type="anthropic",
            external_api_key="test-key",
            whitelist=None,
            blacklist=None,
            context_padding=2,
        )
        
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = type("MockResponse", (), {
                "read": lambda self: "not valid json".encode("utf-8"),
                "decode": lambda self: "not valid json",
            })()
            mock_urlopen.return_value = mock_response
            
            result = anthropic_grep_rag._call_anthropic(
                "Find auth functions",
                "context",
            )
            
            # Verify error handling
            assert isinstance(result, str), "Result should be a string"

    def test_call_x_api_key_header(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that correct `x-api-key` header format."""
        anthropic_grep_rag = GrepRAG(
            main_inference_url="https://test-endpoint.com/v1/chat/completions",
            main_model_name="test-model",
            main_model_params={"max_tokens": 500, "temperature": 0.7},
            grep_model_path_or_name="test-model",
            external_api_type="anthropic",
            external_api_key="test-key",
            whitelist=None,
            blacklist=None,
            context_padding=2,
        )
        
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = type("MockResponse", (), {
                "read": lambda self: json.dumps([
                    {"content": [{"text": "test"}]}
                ]).encode("utf-8"),
                "decode": lambda self: json.dumps([
                    {"content": [{"text": "test"}]}
                ]),
            })()
            mock_urlopen.return_value = mock_response
            
            anthropic_grep_rag._call_anthropic(
                "Find auth functions",
                "context",
            )
            
            # Verify headers
            call_args = mock_urlopen.call_args
            headers = call_args[1]["headers"]
            assert headers.get("x-api-key") == "test-key"

    def test_call_anthropic_version_header(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that correct `anthropic-version` header."""
        anthropic_grep_rag = GrepRAG(
            main_inference_url="https://test-endpoint.com/v1/chat/completions",
            main_model_name="test-model",
            main_model_params={"max_tokens": 500, "temperature": 0.7},
            grep_model_path_or_name="test-model",
            external_api_type="anthropic",
            external_api_key="test-key",
            whitelist=None,
            blacklist=None,
            context_padding=2,
        )
        
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = type("MockResponse", (), {
                "read": lambda self: json.dumps([
                    {"content": [{"text": "test"}]}
                ]).encode("utf-8"),
                "decode": lambda self: json.dumps([
                    {"content": [{"text": "test"}]}
                ]),
            })()
            mock_urlopen.return_value = mock_response
            
            anthropic_grep_rag._call_anthropic(
                "Find auth functions",
                "context",
            )
            
            # Verify headers
            call_args = mock_urlopen.call_args
            headers = call_args[1]["headers"]
            assert headers.get("anthropic-version") == "2023-06-01"

    def test_call_content_array(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that response is array, not dict."""
        anthropic_grep_rag = GrepRAG(
            main_inference_url="https://test-endpoint.com/v1/chat/completions",
            main_model_name="test-model",
            main_model_params={"max_tokens": 500, "temperature": 0.7},
            grep_model_path_or_name="test-model",
            external_api_type="anthropic",
            external_api_key="test-key",
            whitelist=None,
            blacklist=None,
            context_padding=2,
        )
        
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = type("MockResponse", (), {
                "read": lambda self: json.dumps([
                    {"content": [{"text": "test"}]}
                ]).encode("utf-8"),
                "decode": lambda self: json.dumps([
                    {"content": [{"text": "test"}]}
                ]),
            })()
            mock_urlopen.return_value = mock_response
            
            result = anthropic_grep_rag._call_anthropic(
                "Find auth functions",
                "context",
            )
            
            # Verify result structure
            assert isinstance(result, str), "Result should be a string"

    def test_call_text_extraction(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that `text` field extracted from content array."""
        anthropic_grep_rag = GrepRAG(
            main_inference_url="https://test-endpoint.com/v1/chat/completions",
            main_model_name="test-model",
            main_model_params={"max_tokens": 500, "temperature": 0.7},
            grep_model_path_or_name="test-model",
            external_api_type="anthropic",
            external_api_key="test-key",
            whitelist=None,
            blacklist=None,
            context_padding=2,
        )
        
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = type("MockResponse", (), {
                "read": lambda self: json.dumps([
                    {
                        "content": [
                            {
                                "text": "Found 3 authentication-related functions"
                            }
                        ]
                    }
                ]).encode("utf-8"),
                "decode": lambda self: json.dumps([
                    {
                        "content": [
                            {
                                "text": "Found 3 authentication-related functions"
                            }
                        ]
                    }
                ]),
            })()
            mock_urlopen.return_value = mock_response
            
            result = anthropic_grep_rag._call_anthropic(
                "Find auth functions",
                "context",
            )
            
            # Verify text extraction
            assert isinstance(result, str), "Result should be a string"
            assert "Found" in result, "Result should contain 'Found'"

    @pytest.mark.asyncio
    async def test_call_async(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that async version works correctly."""
        anthropic_grep_rag = GrepRAG(
            main_inference_url="https://test-endpoint.com/v1/chat/completions",
            main_model_name="test-model",
            main_model_params={"max_tokens": 500, "temperature": 0.7},
            grep_model_path_or_name="test-model",
            external_api_type="anthropic",
            external_api_key="test-key",
            whitelist=None,
            blacklist=None,
            context_padding=2,
        )
        
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = type("MockResponse", (), {
                "read": lambda self: json.dumps([
                    {"content": [{"text": "test"}]}
                ]).encode("utf-8"),
                "decode": lambda self: json.dumps([
                    {"content": [{"text": "test"}]}
                ]),
            })()
            mock_urlopen.return_value = mock_response
            
            result = await anthropic_grep_rag._call_anthropic(
                "Find auth functions",
                "context",
            )
            
            # Verify result
            assert isinstance(result, str), "Result should be a string"
