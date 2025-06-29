"""Configuration management for Simpulse.

This module provides configuration classes and utilities for managing
Simpulse settings, including Claude Code CLI integration and optimization
parameters.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from .claude.claude_code_client import ClaudeBackend
from .evolution.models import OptimizationGoal

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class ClaudeConfig:
    """Configuration for Claude Code CLI integration."""
    backend: ClaudeBackend = ClaudeBackend.CLAUDE_CODE
    command_path: str = "claude"
    timeout: int = 120
    retry_attempts: int = 3
    cache_responses: bool = True
    cache_dir: Optional[str] = None
    max_context_length: int = 100000
    
    def __post_init__(self):
        """Set default cache directory."""
        if self.cache_responses and not self.cache_dir:
            self.cache_dir = str(Path.home() / ".simpulse" / "cache")


@dataclass 
class ProfilingConfig:
    """Configuration for Lean profiling."""
    timeout: float = 300.0
    trace_flags: List[str] = field(default_factory=lambda: [
        "profiler.output", 
        "Meta.Tactic.simp"
    ])
    lake_path: str = "lake"
    lean_path: str = "lean"
    output_format: str = "json"  # json or text
    collect_memory_stats: bool = True
    
    
@dataclass
class OptimizationConfig:
    """Configuration for optimization process."""
    goal: OptimizationGoal = OptimizationGoal.MINIMIZE_TIME
    max_mutations_per_session: int = 50
    max_parallel_evaluations: int = 4
    confidence_threshold: float = 0.6
    improvement_threshold: float = 0.05  # 5% minimum improvement
    safety_checks: bool = True
    backup_original_files: bool = True
    
    # Evolution parameters
    population_size: int = 20
    generations: int = 10
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    elite_size: int = 5


@dataclass
class PathConfig:
    """Configuration for file paths."""
    project_root: Optional[Path] = None
    source_dirs: List[Path] = field(default_factory=list)
    output_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    log_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Set default paths."""
        if not self.project_root:
            self.project_root = Path.cwd()
            
        if not self.output_dir:
            self.output_dir = self.project_root / "simpulse_output"
            
        if not self.cache_dir:
            self.cache_dir = Path.home() / ".simpulse" / "cache"
            
        if not self.log_dir:
            self.log_dir = self.project_root / "logs"
            
        # Convert strings to Path objects
        self.project_root = Path(self.project_root)
        self.output_dir = Path(self.output_dir)
        self.cache_dir = Path(self.cache_dir)
        self.log_dir = Path(self.log_dir)
        self.source_dirs = [Path(d) for d in self.source_dirs]


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: LogLevel = LogLevel.INFO
    file_logging: bool = True
    console_logging: bool = True
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class Config:
    """Main configuration class for Simpulse."""
    
    DEFAULT_CONFIG_FILE = "simpulse.json"
    CONFIG_ENV_VAR = "SIMPULSE_CONFIG"
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """Initialize configuration.
        
        Args:
            config_file: Path to configuration file
        """
        self.claude = ClaudeConfig()
        self.profiling = ProfilingConfig()
        self.optimization = OptimizationConfig()
        self.paths = PathConfig()
        self.logging = LoggingConfig()
        
        # Load configuration
        self._config_file = self._resolve_config_file(config_file)
        self.load_from_file(self._config_file)
        self._apply_environment_overrides()
        self._validate_config()
        
    def _resolve_config_file(self, config_file: Optional[Union[str, Path]]) -> Optional[Path]:
        """Resolve configuration file path.
        
        Args:
            config_file: Provided config file path
            
        Returns:
            Resolved Path or None
        """
        # Priority order:
        # 1. Explicitly provided file
        # 2. Environment variable
        # 3. Default file in current directory
        # 4. Default file in home directory
        
        if config_file:
            return Path(config_file)
            
        env_config = os.getenv(self.CONFIG_ENV_VAR)
        if env_config:
            return Path(env_config)
            
        # Check current directory
        current_config = Path.cwd() / self.DEFAULT_CONFIG_FILE
        if current_config.exists():
            return current_config
            
        # Check home directory
        home_config = Path.home() / ".simpulse" / self.DEFAULT_CONFIG_FILE
        if home_config.exists():
            return home_config
            
        return None
        
    def load_from_file(self, config_file: Optional[Path] = None) -> bool:
        """Load configuration from file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            True if file was loaded successfully
        """
        if not config_file or not config_file.exists():
            logger.info("No configuration file found, using defaults")
            return False
            
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
                
            self._update_from_dict(data)
            logger.info(f"Loaded configuration from {config_file}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load configuration from {config_file}: {e}")
            return False
            
    def save_to_file(self, config_file: Optional[Path] = None) -> bool:
        """Save configuration to file.
        
        Args:
            config_file: Path to save configuration
            
        Returns:
            True if saved successfully
        """
        if not config_file:
            config_file = self._config_file or Path.cwd() / self.DEFAULT_CONFIG_FILE
            
        try:
            # Ensure directory exists
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
                
            logger.info(f"Saved configuration to {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_file}: {e}")
            return False
            
    def _update_from_dict(self, data: Dict[str, Any]):
        """Update configuration from dictionary.
        
        Args:
            data: Configuration data
        """
        # Update each section
        if "claude" in data:
            self._update_dataclass(self.claude, data["claude"])
            
        if "profiling" in data:
            self._update_dataclass(self.profiling, data["profiling"])
            
        if "optimization" in data:
            self._update_dataclass(self.optimization, data["optimization"])
            
        if "paths" in data:
            self._update_dataclass(self.paths, data["paths"])
            
        if "logging" in data:
            self._update_dataclass(self.logging, data["logging"])
            
    def _update_dataclass(self, obj: Any, data: Dict[str, Any]):
        """Update dataclass fields from dictionary.
        
        Args:
            obj: Dataclass object to update
            data: Dictionary with new values
        """
        for key, value in data.items():
            if hasattr(obj, key):
                # Handle enum fields
                field_type = type(getattr(obj, key))
                if hasattr(field_type, '__members__'):  # Enum
                    if isinstance(value, str):
                        try:
                            value = field_type(value)
                        except ValueError:
                            logger.warning(f"Invalid enum value {value} for {key}")
                            continue
                            
                setattr(obj, key, value)
                
    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        # Claude configuration
        if os.getenv("SIMPULSE_CLAUDE_COMMAND"):
            self.claude.command_path = os.getenv("SIMPULSE_CLAUDE_COMMAND")
            
        if os.getenv("SIMPULSE_CLAUDE_TIMEOUT"):
            try:
                self.claude.timeout = int(os.getenv("SIMPULSE_CLAUDE_TIMEOUT"))
            except ValueError:
                pass
                
        # Profiling configuration
        if os.getenv("SIMPULSE_LAKE_PATH"):
            self.profiling.lake_path = os.getenv("SIMPULSE_LAKE_PATH")
            
        if os.getenv("SIMPULSE_LEAN_PATH"):
            self.profiling.lean_path = os.getenv("SIMPULSE_LEAN_PATH")
            
        # Logging level
        if os.getenv("SIMPULSE_LOG_LEVEL"):
            try:
                self.logging.level = LogLevel(os.getenv("SIMPULSE_LOG_LEVEL").upper())
            except ValueError:
                pass
                
    def _validate_config(self):
        """Validate configuration values."""
        # Validate timeouts
        if self.claude.timeout <= 0:
            logger.warning("Invalid Claude timeout, using default")
            self.claude.timeout = 120
            
        if self.profiling.timeout <= 0:
            logger.warning("Invalid profiling timeout, using default")
            self.profiling.timeout = 300.0
            
        # Validate thresholds
        if not 0 <= self.optimization.confidence_threshold <= 1:
            logger.warning("Invalid confidence threshold, using default")
            self.optimization.confidence_threshold = 0.6
            
        if self.optimization.improvement_threshold < 0:
            logger.warning("Invalid improvement threshold, using default")
            self.optimization.improvement_threshold = 0.05
            
        # Create directories if needed
        for path in [self.paths.output_dir, self.paths.cache_dir, self.paths.log_dir]:
            if path:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to create directory {path}: {e}")
                    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {
            "claude": asdict(self.claude),
            "profiling": asdict(self.profiling),
            "optimization": asdict(self.optimization),
            "paths": asdict(self.paths),
            "logging": asdict(self.logging)
        }
        
    def setup_logging(self):
        """Setup logging based on configuration."""
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.logging.level.value)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # Setup formatters
        formatter = logging.Formatter(self.logging.format)
        
        # Console handler
        if self.logging.console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
            
        # File handler
        if self.logging.file_logging and self.paths.log_dir:
            try:
                from logging.handlers import RotatingFileHandler
                
                log_file = self.paths.log_dir / "simpulse.log"
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=self.logging.max_file_size,
                    backupCount=self.logging.backup_count
                )
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
                
            except Exception as e:
                logger.warning(f"Failed to setup file logging: {e}")
                
    def get_cache_path(self, cache_type: str) -> Path:
        """Get cache path for specific cache type.
        
        Args:
            cache_type: Type of cache (e.g., 'claude', 'profiling')
            
        Returns:
            Path to cache directory
        """
        cache_path = self.paths.cache_dir / cache_type
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path
        
    def is_claude_available(self) -> bool:
        """Check if Claude Code CLI is available.
        
        Returns:
            True if Claude is available
        """
        from .claude.claude_code_client import ClaudeCodeClient
        
        client = ClaudeCodeClient(command=self.claude.command_path)
        return client._validate_claude_installation()
    
    def _validate_api_key(self, key: str) -> None:
        """Validate API key format for security.
        
        Args:
            key: API key to validate
            
        Raises:
            ValueError: If key is invalid
        """
        from .security.validators import validate_api_key
        validate_api_key(key)


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance.
    
    Returns:
        Global Config instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config):
    """Set global configuration instance.
    
    Args:
        config: New configuration instance
    """
    global _global_config
    _global_config = config
    
    
def load_config(config_file: Optional[Union[str, Path]] = None) -> Config:
    """Load and set global configuration.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Loaded configuration
    """
    config = Config(config_file)
    set_config(config)
    return config