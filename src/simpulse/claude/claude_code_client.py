"""Claude Code CLI client for Simpulse.

This module provides integration with Claude Code CLI for generating
intelligent simp rule mutations and optimizations.
"""

import asyncio
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ClaudeBackend(Enum):
    """Available Claude backends."""
    CLAUDE_CODE = "claude_code"  # Via CLI subprocess
    API = "api"  # Traditional API (optional)


@dataclass
class ClaudeResponse:
    """Response from Claude interaction."""
    content: str
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0
    backend: ClaudeBackend = ClaudeBackend.CLAUDE_CODE


class ClaudeCodeClient:
    """Client for interacting with Claude Code CLI."""
    
    def __init__(self, 
                 command: str = "claude",
                 timeout: int = 120,
                 retry_attempts: int = 3,
                 working_dir: Optional[Path] = None):
        """Initialize Claude Code client.
        
        Args:
            command: Claude Code CLI command name or path
            timeout: Command execution timeout in seconds
            retry_attempts: Number of retry attempts on failure
            working_dir: Working directory for Claude operations
        """
        self.command = command
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.working_dir = working_dir or Path.cwd()
        self._available = None
        self._rate_limiter = None
        self._last_call_time = None
        
    def _validate_claude_installation(self) -> bool:
        """Check if Claude Code CLI is available.
        
        Returns:
            True if Claude Code is available, False otherwise
        """
        if self._available is not None:
            return self._available
            
        try:
            result = shutil.which(self.command)
            if result is None:
                logger.warning(f"Claude Code CLI '{self.command}' not found in PATH")
                self._available = False
                return False
                
            # Test version command
            process = asyncio.create_subprocess_exec(
                self.command, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # This is a sync check, so we'll use subprocess for simplicity
            import subprocess
            result = subprocess.run(
                [self.command, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info(f"Claude Code CLI available: {result.stdout.strip()}")
                self._available = True
                return True
            else:
                logger.warning(f"Claude Code CLI check failed: {result.stderr}")
                self._available = False
                return False
                
        except Exception as e:
            logger.warning(f"Failed to validate Claude Code installation: {e}")
            self._available = False
            return False
            
    async def query_claude(self, 
                          prompt: str, 
                          save_context: bool = True,
                          project_context: Optional[str] = None) -> ClaudeResponse:
        """Execute Claude Code CLI with prompt.
        
        Args:
            prompt: The prompt to send to Claude
            save_context: Whether to save context for follow-up queries
            project_context: Additional project context to include
            
        Returns:
            ClaudeResponse with result
        """
        if not self._validate_claude_installation():
            return ClaudeResponse(
                content="",
                success=False,
                error="Claude Code CLI not available",
                backend=ClaudeBackend.CLAUDE_CODE
            )
        
        start_time = time.time()
        
        # Create temporary file for prompt
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            # Build complete prompt with context
            full_prompt = self._build_full_prompt(prompt, project_context)
            f.write(full_prompt)
            prompt_file = Path(f.name)
            
        try:
            for attempt in range(self.retry_attempts):
                try:
                    # Build command
                    cmd = self._build_command(prompt_file, save_context)
                    
                    logger.debug(f"Executing Claude Code CLI (attempt {attempt + 1}): {' '.join(cmd)}")
                    
                    # Execute command
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=self.working_dir
                    )
                    
                    try:
                        stdout_data, stderr_data = await asyncio.wait_for(
                            process.communicate(),
                            timeout=self.timeout
                        )
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                        raise TimeoutError(f"Claude Code CLI timed out after {self.timeout} seconds")
                    
                    stdout = stdout_data.decode('utf-8', errors='replace')
                    stderr = stderr_data.decode('utf-8', errors='replace')
                    
                    execution_time = time.time() - start_time
                    
                    if process.returncode == 0:
                        # Parse and clean response
                        content = self._parse_response(stdout)
                        
                        return ClaudeResponse(
                            content=content,
                            success=True,
                            execution_time=execution_time,
                            backend=ClaudeBackend.CLAUDE_CODE
                        )
                    else:
                        error_msg = f"Claude Code CLI failed (exit {process.returncode}): {stderr}"
                        logger.warning(error_msg)
                        
                        if attempt == self.retry_attempts - 1:
                            return ClaudeResponse(
                                content="",
                                success=False,
                                error=error_msg,
                                execution_time=execution_time,
                                backend=ClaudeBackend.CLAUDE_CODE
                            )
                        
                        # Wait before retry
                        await asyncio.sleep(2 ** attempt)
                        
                except Exception as e:
                    logger.error(f"Error executing Claude Code CLI (attempt {attempt + 1}): {e}")
                    
                    if attempt == self.retry_attempts - 1:
                        return ClaudeResponse(
                            content="",
                            success=False,
                            error=str(e),
                            execution_time=time.time() - start_time,
                            backend=ClaudeBackend.CLAUDE_CODE
                        )
                    
                    await asyncio.sleep(2 ** attempt)
                    
        finally:
            # Cleanup temporary file
            if prompt_file.exists():
                prompt_file.unlink()
                
        return ClaudeResponse(
            content="",
            success=False,
            error="All retry attempts failed",
            execution_time=time.time() - start_time,
            backend=ClaudeBackend.CLAUDE_CODE
        )
        
    def _build_command(self, prompt_file: Path, save_context: bool = True) -> List[str]:
        """Build Claude Code CLI command.
        
        Args:
            prompt_file: Path to prompt file
            save_context: Whether to save context
            
        Returns:
            Command as list of strings
        """
        cmd = [self.command, "-p", str(prompt_file)]
        
        # Add additional flags if needed
        if not save_context:
            # Add flag to not save context if supported
            # This depends on Claude Code CLI features
            pass
            
        return cmd
        
    def _build_full_prompt(self, prompt: str, project_context: Optional[str] = None) -> str:
        """Build complete prompt with context.
        
        Args:
            prompt: Main prompt content
            project_context: Additional project context
            
        Returns:
            Complete prompt string
        """
        parts = []
        
        # Add project context if provided
        if project_context:
            parts.append("## Project Context")
            parts.append(project_context)
            parts.append("")
        
        # Add main prompt
        parts.append("## Task")
        parts.append(prompt)
        
        return "\n".join(parts)
        
    def _parse_response(self, stdout: str) -> str:
        """Parse response from Claude Code CLI.
        
        Args:
            stdout: Raw stdout from command
            
        Returns:
            Cleaned response content
        """
        # Claude Code CLI might include formatting or metadata
        # For now, return the raw output, but this could be enhanced
        # to extract specific sections or clean formatting
        
        lines = stdout.split('\n')
        content_lines = []
        
        # Skip any CLI output/formatting
        for line in lines:
            # Skip lines that look like CLI output
            if line.startswith('[') and line.endswith(']'):
                continue
            if line.startswith('claude:'):
                continue
                
            content_lines.append(line)
            
        return '\n'.join(content_lines).strip()
        
    async def suggest_mutations(self, 
                               profile_data: Dict[str, Any],
                               rules: List[Dict[str, Any]], 
                               top_k: int = 5,
                               context: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate mutation suggestions based on profile data and rules.
        
        Args:
            profile_data: Performance profiling data
            rules: List of simp rules with metadata
            top_k: Number of top suggestions to return
            context: Additional context about the optimization goals
            
        Returns:
            List of mutation suggestions
        """
        from .prompt_builder import PromptBuilder
        from .response_parser import ResponseParser
        
        # Build mutation prompt
        prompt_builder = PromptBuilder()
        prompt = prompt_builder.build_mutation_prompt(
            profile_data=profile_data,
            rules=rules,
            top_k=top_k,
            context=context
        )
        
        # Query Claude
        response = await self.query_claude(prompt, save_context=True)
        
        if not response.success:
            logger.error(f"Failed to get mutation suggestions: {response.error}")
            return []
            
        # Parse suggestions
        parser = ResponseParser()
        try:
            suggestions = parser.parse_mutations(response.content)
            logger.info(f"Generated {len(suggestions)} mutation suggestions")
            return suggestions
        except Exception as e:
            logger.error(f"Failed to parse mutation suggestions: {e}")
            return []
            
    async def analyze_performance_bottlenecks(self, 
                                            profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance bottlenecks using Claude.
        
        Args:
            profile_data: Performance profiling data
            
        Returns:
            Analysis results with recommendations
        """
        from .prompt_builder import PromptBuilder
        
        prompt_builder = PromptBuilder()
        prompt = prompt_builder.build_analysis_prompt(profile_data)
        
        response = await self.query_claude(prompt, save_context=False)
        
        if not response.success:
            logger.error(f"Failed to analyze performance: {response.error}")
            return {"success": False, "error": response.error}
            
        return {
            "success": True,
            "analysis": response.content,
            "execution_time": response.execution_time
        }
        
    async def explain_optimization(self, 
                                  original_rule: str,
                                  optimized_rule: str,
                                  performance_improvement: Dict[str, Any]) -> str:
        """Get explanation for a specific optimization.
        
        Args:
            original_rule: Original simp rule
            optimized_rule: Optimized version
            performance_improvement: Measured improvement data
            
        Returns:
            Explanation of the optimization
        """
        prompt = f"""
Explain this Lean 4 simp rule optimization:

## Original Rule
```lean
{original_rule}
```

## Optimized Rule  
```lean
{optimized_rule}
```

## Performance Improvement
{json.dumps(performance_improvement, indent=2)}

Please explain:
1. What changed and why
2. How this improves performance
3. Any potential trade-offs or considerations
4. Whether this optimization pattern could apply elsewhere

Keep the explanation concise but technical.
"""

        response = await self.query_claude(prompt, save_context=False)
        
        if response.success:
            return response.content
        else:
            return f"Failed to generate explanation: {response.error}"
    
    def _check_rate_limit(self) -> bool:
        """Check if rate limited.
        
        Returns:
            True if rate limited, False if OK to proceed
        """
        from ..security.validators import RateLimiter
        
        if self._rate_limiter is None:
            self._rate_limiter = RateLimiter(max_calls=60, window_seconds=60)
        
        # Update last call time if not rate limited
        limited = self._rate_limiter.check_rate_limit()
        if not limited:
            self._last_call_time = time.time()
            
        return limited