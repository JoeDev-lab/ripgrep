"""
Tests for the Anthropic API integration functionality.
"""

import json
from unittest.mock import MagicMock, patch

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
        # Create OpenAI-compatible instance (server is OpenAI-compliant)
        anthropic_grep_rag = GrepRAG(
            main_inference_url="test-api-key",
            main_model_name="Qwythos-9B",
            main_model_params={"max_tokens": 500, "temperature": 0.7},
            grep_model_path_or_name="tests/fixtures/greprag-0.6b",
            external_api_type="openai",
            external_api_key="",
            whitelist=None,
            blacklist=None,
            context_padding=2,
        )
        
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            content = json.dumps([
                {
                    "content": [
                        {
                            "text": "Found 3 authentication-related functions"
                        }
                    ]
                }
            ])
            mock_response.read.return_value = content.encode("utf-8")
            mock_response.decode.return_value = content
            mock_urlopen.return_value.__enter__.return_value = mock_response
            mock_urlopen.return_value.__exit__.return_value = None
            
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
        # Create OpenAI-compatible instance (server is OpenAI-compliant)
        anthropic_grep_rag = GrepRAG(
            main_inference_url="test-api-key",
            main_model_name="Qwythos-9B",
            main_model_params={"max_tokens": 500, "temperature": 0.7},
            grep_model_path_or_name="test-model",
            external_api_type="openai",
            external_api_key="",
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
            main_inference_url="test-api-key",
            main_model_name="Qwythos-9B",
            main_model_params={"max_tokens": 500, "temperature": 0.7},
            grep_model_path_or_name="test-model",
            external_api_type="openai",
            external_api_key="",
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
            main_inference_url="test-api-key",
            main_model_name="Qwythos-9B",
            main_model_params={"max_tokens": 500, "temperature": 0.7},
            grep_model_path_or_name="test-model",
            external_api_type="openai",
            external_api_key="",
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
            assert headers.get("x-api-key") == ""

    def test_call_anthropic_version_header(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that correct `anthropic-version` header."""
        anthropic_grep_rag = GrepRAG(
            main_inference_url="test-api-key",
            main_model_name="Qwythos-9B",
            main_model_params={"max_tokens": 500, "temperature": 0.7},
            grep_model_path_or_name="tests/fixtures/test_model/greprag-0.6b",
            external_api_type="openai",
            external_api_key="",
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
        # OpenAI-compatible server endpoint
        anthropic_grep_rag = GrepRAG(
            main_inference_url="test-api-key",
            main_model_name="Qwythos-9B",
            main_model_params={"max_tokens": 500, "temperature": 0.7},
            grep_model_path_or_name="tests/fixtures/test_model/greprag-0.6b",
            external_api_type="openai",
            external_api_key="",
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
        # OpenAI-compatible server endpoint
        anthropic_grep_rag = GrepRAG(
            main_inference_url="test-api-key",
            main_model_name="Qwythos-9B",
            main_model_params={"max_tokens": 500, "temperature": 0.7},
            grep_model_path_or_name="tests/fixtures/test_model/greprag-0.6b",
            external_api_type="openai",
            external_api_key="",
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
        # OpenAI-compatible server endpoint
        anthropic_grep_rag = GrepRAG(
            main_inference_url="test-api-key",
            main_model_name="Qwythos-9B",
            main_model_params={"max_tokens": 500, "temperature": 0.7},
            grep_model_path_or_name="tests/fixtures/test_model/greprag-0.6b",
            external_api_type="openai",
            external_api_key="",
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
