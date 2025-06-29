"""
Tests for Claude Code Client module.
"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from simpulse.claude.claude_code_client import ClaudeCodeClient, ClaudeResponse


class TestClaudeCodeClient:
    """Test suite for ClaudeCodeClient."""

    @pytest.fixture
    def claude_client(self):
        """Create a Claude client instance for testing."""
        return ClaudeCodeClient(
            claude_executable="claude",
            timeout_seconds=5,
            max_retries=2,
            save_context=False,
        )

    def test_initialization(self, claude_client):
        """Test client initialization."""
        assert claude_client.claude_executable == "claude"
        assert claude_client.timeout_seconds == 5
        assert claude_client.max_retries == 2
        assert not claude_client.save_context

    def test_is_available_success(self, claude_client):
        """Test checking Claude availability when available."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="Claude Code CLI v1.0.0")

            available = claude_client.is_available()

            assert available
            mock_run.assert_called_once_with(
                ["claude", "--version"], capture_output=True, text=True, timeout=5
            )

    def test_is_available_not_found(self, claude_client):
        """Test checking Claude availability when not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            available = claude_client.is_available()
            assert not available

    @pytest.mark.asyncio
    async def test_query_claude_success(self, claude_client):
        """Test successful Claude query."""
        mock_response = {
            "content": "Here is my response to your prompt",
            "tokens_used": 150,
            "model": "claude-3-opus",
        }

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            # Mock temp file operations
            mock_file = Mock()
            mock_file.name = "/tmp/test_prompt.txt"
            mock_temp.return_value.__enter__.return_value = mock_file

            with patch("asyncio.create_subprocess_exec") as mock_subprocess:
                # Mock subprocess execution
                mock_process = AsyncMock()
                mock_process.returncode = 0
                mock_process.communicate = AsyncMock(
                    return_value=(json.dumps(mock_response).encode(), b"")
                )
                mock_subprocess.return_value = mock_process

                # Execute query
                response = await claude_client.query_claude("Test prompt")

                # Verify response
                assert isinstance(response, ClaudeResponse)
                assert response.success
                assert response.content == "Here is my response to your prompt"
                assert response.tokens_used == 150

    @pytest.mark.asyncio
    async def test_query_claude_timeout(self, claude_client):
        """Test Claude query timeout handling."""
        with patch("tempfile.NamedTemporaryFile"):
            with patch("asyncio.create_subprocess_exec") as mock_subprocess:
                # Mock timeout
                mock_subprocess.side_effect = asyncio.TimeoutError()

                response = await claude_client.query_claude("Test prompt")

                # Should handle timeout gracefully
                assert not response.success
                assert "timeout" in response.content.lower()

    @pytest.mark.asyncio
    async def test_query_claude_retry(self, claude_client):
        """Test retry logic on failure."""
        claude_client.max_retries = 2

        with patch("tempfile.NamedTemporaryFile"):
            with patch("asyncio.create_subprocess_exec") as mock_subprocess:
                # First call fails, second succeeds
                mock_process_fail = AsyncMock()
                mock_process_fail.returncode = 1
                mock_process_fail.communicate = AsyncMock(return_value=(b"", b"Error"))

                mock_process_success = AsyncMock()
                mock_process_success.returncode = 0
                mock_process_success.communicate = AsyncMock(
                    return_value=(
                        json.dumps({"content": "Success after retry"}).encode(),
                        b"",
                    )
                )

                mock_subprocess.side_effect = [mock_process_fail, mock_process_success]

                response = await claude_client.query_claude("Test prompt")

                # Should succeed after retry
                assert response.success
                assert response.content == "Success after retry"
                assert mock_subprocess.call_count == 2

    @pytest.mark.asyncio
    async def test_query_claude_with_context(self, claude_client):
        """Test Claude query with context saving."""
        claude_client.save_context = True

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            with patch("asyncio.create_subprocess_exec") as mock_subprocess:
                # Setup mocks
                mock_file = Mock()
                mock_file.name = "/tmp/test_context.txt"
                mock_temp.return_value.__enter__.return_value = mock_file

                mock_process = AsyncMock()
                mock_process.returncode = 0
                mock_process.communicate = AsyncMock(
                    return_value=(
                        json.dumps({"content": "Response with context"}).encode(),
                        b"",
                    )
                )
                mock_subprocess.return_value = mock_process

                await claude_client.query_claude("Test prompt")

                # Should save context
                assert "--save-context" in str(mock_subprocess.call_args)

    @pytest.mark.asyncio
    async def test_parse_response_json(self, claude_client):
        """Test parsing JSON response."""
        json_output = json.dumps(
            {"content": "Test response", "tokens_used": 100, "model": "claude-3"}
        )

        response = claude_client._parse_response(json_output, "prompt", 0.5)

        assert response.content == "Test response"
        assert response.tokens_used == 100
        assert response.execution_time == 0.5

    def test_parse_response_plain_text(self, claude_client):
        """Test parsing plain text response."""
        plain_output = "This is a plain text response"

        response = claude_client._parse_response(plain_output, "prompt", 0.3)

        assert response.content == "This is a plain text response"
        assert response.tokens_used == 0  # Unknown for plain text
        assert response.execution_time == 0.3

    def test_parse_response_invalid_json(self, claude_client):
        """Test parsing invalid JSON response."""
        invalid_json = '{"content": "Incomplete JSON'

        response = claude_client._parse_response(invalid_json, "prompt", 0.2)

        # Should fall back to plain text
        assert response.content == invalid_json
        assert response.tokens_used == 0

    @pytest.mark.asyncio
    async def test_error_handling_subprocess_error(self, claude_client):
        """Test handling of subprocess errors."""
        with patch("tempfile.NamedTemporaryFile"):
            with patch(
                "asyncio.create_subprocess_exec",
                side_effect=Exception("Subprocess failed"),
            ):
                response = await claude_client.query_claude("Test prompt")

                assert not response.success
                assert "error" in response.content.lower()

    @pytest.mark.asyncio
    async def test_large_prompt_handling(self, claude_client):
        """Test handling of large prompts."""
        large_prompt = "x" * 100000  # 100k characters

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_file = Mock()
            mock_temp.return_value.__enter__.return_value = mock_file

            with patch("asyncio.create_subprocess_exec") as mock_subprocess:
                mock_process = AsyncMock()
                mock_process.returncode = 0
                mock_process.communicate = AsyncMock(
                    return_value=(
                        json.dumps({"content": "Handled large prompt"}).encode(),
                        b"",
                    )
                )
                mock_subprocess.return_value = mock_process

                response = await claude_client.query_claude(large_prompt)

                # Should handle large prompts via file
                assert response.success
                mock_file.write.assert_called()

    def test_validate_executable_path(self, claude_client):
        """Test validation of executable path."""
        # Test with absolute path
        claude_client.claude_executable = "/usr/local/bin/claude"
        with patch("os.path.exists", return_value=True):
            with patch("subprocess.run", return_value=Mock(returncode=0)):
                assert claude_client.is_available()

        # Test with command in PATH
        claude_client.claude_executable = "claude"
        with patch("subprocess.run", return_value=Mock(returncode=0)):
            assert claude_client.is_available()


class TestClaudeResponse:
    """Test suite for ClaudeResponse dataclass."""

    def test_claude_response_creation(self):
        """Test creating ClaudeResponse instances."""
        response = ClaudeResponse(
            content="Test content",
            success=True,
            tokens_used=50,
            execution_time=1.5,
            model="claude-3",
            cached=False,
        )

        assert response.content == "Test content"
        assert response.success
        assert response.tokens_used == 50
        assert response.execution_time == 1.5
        assert response.model == "claude-3"
        assert not response.cached

    def test_claude_response_defaults(self):
        """Test ClaudeResponse default values."""
        response = ClaudeResponse(
            content="Test", success=True, tokens_used=0, execution_time=0.0
        )

        assert response.model is None
        assert not response.cached


@pytest.mark.integration
class TestClaudeIntegration:
    """Integration tests for Claude client."""

    @pytest.mark.requires_claude
    @pytest.mark.asyncio
    async def test_real_claude_query(self):
        """Test with real Claude CLI if available."""
        client = ClaudeCodeClient()

        if not client.is_available():
            pytest.skip("Claude Code CLI not available")

        response = await client.query_claude("Say 'Hello, test!'")

        assert response.success
        assert "hello" in response.content.lower() or "test" in response.content.lower()
