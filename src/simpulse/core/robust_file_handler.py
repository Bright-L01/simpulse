"""
Robust file handling system for Simpulse.

Provides comprehensive file operations with encoding detection, corruption handling,
memory management, and graceful error recovery.
"""

import hashlib
import mimetypes
import os
import shutil
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import chardet

    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

try:
    import magic

    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False

from ..errors import ErrorCategory, ErrorContext, ErrorHandler, ErrorSeverity
from .graceful_degradation import GracefulDegradationManager, PartialResult, ResultStatus


class FileType(Enum):
    """Detected file types."""

    LEAN = "lean"
    TEXT = "text"
    BINARY = "binary"
    JSON = "json"
    MARKDOWN = "markdown"
    CONFIG = "config"
    UNKNOWN = "unknown"


class EncodingConfidence(Enum):
    """Confidence levels for encoding detection."""

    HIGH = "high"  # >90% confidence
    MEDIUM = "medium"  # 70-90% confidence
    LOW = "low"  # 50-70% confidence
    GUESS = "guess"  # <50% confidence


@dataclass
class FileInfo:
    """Comprehensive file information."""

    path: Path
    size: int
    file_type: FileType
    encoding: Optional[str]
    encoding_confidence: EncodingConfidence
    is_text: bool
    is_binary: bool
    mime_type: Optional[str]
    checksum: Optional[str]
    created_time: float
    modified_time: float
    is_readable: bool
    is_writable: bool
    potential_corruption: bool
    estimated_processing_time: float


@dataclass
class FileReadResult:
    """Result of file reading operation."""

    content: Optional[Union[str, bytes]]
    file_info: FileInfo
    encoding_used: Optional[str]
    success: bool
    partial_read: bool
    bytes_read: int
    processing_time: float
    warnings: List[str]
    errors: List[str]


class RobustFileHandler:
    """Robust file handler with comprehensive error handling and recovery."""

    def __init__(
        self,
        error_handler: ErrorHandler,
        degradation_manager: GracefulDegradationManager,
        max_file_size_mb: int = 100,
        chunk_size: int = 8192,
        encoding_fallbacks: Optional[List[str]] = None,
    ):
        self.error_handler = error_handler
        self.degradation_manager = degradation_manager
        self.max_file_size_mb = max_file_size_mb
        self.chunk_size = chunk_size
        self.encoding_fallbacks = encoding_fallbacks or ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

        # Cache for file info to avoid repeated analysis
        self.file_info_cache: Dict[str, FileInfo] = {}

        # Statistics
        self.stats = {
            "files_processed": 0,
            "encoding_detections": {},
            "corruption_detected": 0,
            "fallback_encodings_used": 0,
            "large_files_streamed": 0,
        }

    def analyze_file(self, file_path: Path, force_refresh: bool = False) -> FileInfo:
        """Comprehensively analyze a file."""

        cache_key = f"{file_path}_{file_path.stat().st_mtime}"

        if not force_refresh and cache_key in self.file_info_cache:
            return self.file_info_cache[cache_key]

        time.time()

        try:
            stat = file_path.stat()
            size = stat.st_size

            # Basic file type detection
            file_type = self._detect_file_type(file_path)

            # Encoding detection for text files
            encoding = None
            encoding_confidence = EncodingConfidence.GUESS
            is_text = False
            is_binary = False

            if file_type in [
                FileType.LEAN,
                FileType.TEXT,
                FileType.JSON,
                FileType.MARKDOWN,
                FileType.CONFIG,
            ]:
                encoding, encoding_confidence = self._detect_encoding(file_path)
                is_text = True
            else:
                is_binary = True

            # MIME type detection
            mime_type = mimetypes.guess_type(str(file_path))[0]
            if not mime_type and HAS_MAGIC:
                try:
                    mime_type = magic.from_file(str(file_path), mime=True)
                except Exception:
                    pass

            # Calculate checksum for integrity checking
            checksum = self._calculate_checksum(file_path) if size < 50 * 1024 * 1024 else None

            # Check for potential corruption
            potential_corruption = self._check_corruption(file_path, file_type, encoding)

            # Estimate processing time
            estimated_processing_time = self._estimate_processing_time(size, file_type)

            file_info = FileInfo(
                path=file_path,
                size=size,
                file_type=file_type,
                encoding=encoding,
                encoding_confidence=encoding_confidence,
                is_text=is_text,
                is_binary=is_binary,
                mime_type=mime_type,
                checksum=checksum,
                created_time=stat.st_ctime,
                modified_time=stat.st_mtime,
                is_readable=os.access(file_path, os.R_OK),
                is_writable=os.access(file_path, os.W_OK),
                potential_corruption=potential_corruption,
                estimated_processing_time=estimated_processing_time,
            )

            self.file_info_cache[cache_key] = file_info
            return file_info

        except Exception as e:
            context = ErrorContext(operation="file_analysis", file_path=file_path)
            self.error_handler.handle_error(
                category=ErrorCategory.FILE_ACCESS,
                severity=ErrorSeverity.MEDIUM,
                message=f"File analysis failed: {e}",
                context=context,
                exception=e,
            )

            # Return minimal file info
            return FileInfo(
                path=file_path,
                size=0,
                file_type=FileType.UNKNOWN,
                encoding=None,
                encoding_confidence=EncodingConfidence.GUESS,
                is_text=False,
                is_binary=True,
                mime_type=None,
                checksum=None,
                created_time=time.time(),
                modified_time=time.time(),
                is_readable=False,
                is_writable=False,
                potential_corruption=True,
                estimated_processing_time=0.0,
            )

    def read_file_robust(
        self,
        file_path: Path,
        max_size_mb: Optional[int] = None,
        encoding: Optional[str] = None,
        enable_streaming: bool = True,
    ) -> PartialResult[FileReadResult]:
        """Robustly read a file with comprehensive error handling."""

        max_size = (max_size_mb or self.max_file_size_mb) * 1024 * 1024
        start_time = time.time()

        # Analyze file first
        file_info = self.analyze_file(file_path)

        # Check if file is too large
        if file_info.size > max_size:
            if enable_streaming:
                return self._stream_file_read(file_path, file_info, encoding)
            else:
                context = ErrorContext(
                    operation="file_read",
                    file_path=file_path,
                    additional_info={"file_size": file_info.size, "max_size": max_size},
                )

                error = self.error_handler.handle_error(
                    category=ErrorCategory.MEMORY,
                    severity=ErrorSeverity.HIGH,
                    message=f"File too large: {file_info.size / 1024 / 1024:.1f}MB > {max_size / 1024 / 1024:.1f}MB",
                    context=context,
                )

                return PartialResult(
                    data=FileReadResult(
                        content=None,
                        file_info=file_info,
                        encoding_used=None,
                        success=False,
                        partial_read=False,
                        bytes_read=0,
                        processing_time=time.time() - start_time,
                        warnings=[],
                        errors=[error.message],
                    ),
                    status=ResultStatus.FAILED,
                    success_rate=0.0,
                )

        # Check for corruption
        if file_info.potential_corruption:
            f"Potential file corruption detected in {file_path}"
            self.stats["corruption_detected"] += 1

            # Try to read anyway with extra caution
            return self._read_potentially_corrupted_file(file_path, file_info, encoding)

        # Normal file reading
        return self._read_file_normal(file_path, file_info, encoding)

    def _read_file_normal(
        self, file_path: Path, file_info: FileInfo, encoding: Optional[str] = None
    ) -> PartialResult[FileReadResult]:
        """Read file using normal method."""

        start_time = time.time()
        warnings = []
        errors = []

        # Determine encoding
        target_encoding = encoding or file_info.encoding or "utf-8"

        try:
            if file_info.is_text:
                content = self._read_text_file_with_fallback(file_path, target_encoding)
                bytes_read = len(content.encode(target_encoding)) if content else 0
            else:
                with open(file_path, "rb") as f:
                    content = f.read()
                bytes_read = len(content) if content else 0
                target_encoding = None

            result = FileReadResult(
                content=content,
                file_info=file_info,
                encoding_used=target_encoding,
                success=True,
                partial_read=False,
                bytes_read=bytes_read,
                processing_time=time.time() - start_time,
                warnings=warnings,
                errors=errors,
            )

            self.stats["files_processed"] += 1

            return PartialResult(data=result, status=ResultStatus.COMPLETE, success_rate=1.0)

        except Exception as e:
            context = ErrorContext(
                operation="file_read_normal",
                file_path=file_path,
                additional_info={"encoding": target_encoding},
            )

            error = self.error_handler.handle_error(
                category=ErrorCategory.FILE_ACCESS,
                severity=ErrorSeverity.MEDIUM,
                message=f"Normal file read failed: {e}",
                context=context,
                exception=e,
            )

            errors.append(error.message)

            result = FileReadResult(
                content=None,
                file_info=file_info,
                encoding_used=target_encoding,
                success=False,
                partial_read=False,
                bytes_read=0,
                processing_time=time.time() - start_time,
                warnings=warnings,
                errors=errors,
            )

            return PartialResult(data=result, status=ResultStatus.FAILED, success_rate=0.0)

    def _read_potentially_corrupted_file(
        self, file_path: Path, file_info: FileInfo, encoding: Optional[str] = None
    ) -> PartialResult[FileReadResult]:
        """Read file that might be corrupted with extra safety measures."""

        start_time = time.time()
        warnings = ["File shows signs of potential corruption"]
        errors = []

        try:
            # Try to read file in small chunks to isolate corruption
            content_parts = []
            bytes_read = 0
            corruption_detected = False

            with open(file_path, "rb") as f:
                while True:
                    try:
                        chunk = f.read(self.chunk_size)
                        if not chunk:
                            break

                        content_parts.append(chunk)
                        bytes_read += len(chunk)

                        # Check for obvious corruption markers
                        if b"\\x00" in chunk and file_info.is_text:
                            corruption_detected = True
                            warnings.append("Null bytes found in text file")

                    except Exception as chunk_error:
                        corruption_detected = True
                        warnings.append(f"Chunk read error: {chunk_error}")
                        break

            # Combine chunks
            raw_content = b"".join(content_parts)

            # Convert to text if needed
            if file_info.is_text:
                target_encoding = encoding or file_info.encoding or "utf-8"
                try:
                    content = raw_content.decode(target_encoding, errors="replace")

                    # Count replacement characters as corruption indicator
                    replacement_count = content.count("\\ufffd")
                    if replacement_count > 0:
                        warnings.append(f"{replacement_count} invalid characters replaced")

                except Exception:
                    # Try fallback encodings
                    content = self._decode_with_fallbacks(raw_content)
                    warnings.append("Used fallback encoding detection")
            else:
                content = raw_content
                target_encoding = None

            success = not corruption_detected
            status = ResultStatus.DEGRADED if corruption_detected else ResultStatus.COMPLETE
            success_rate = 0.7 if corruption_detected else 1.0

            result = FileReadResult(
                content=content,
                file_info=file_info,
                encoding_used=target_encoding,
                success=success,
                partial_read=corruption_detected,
                bytes_read=bytes_read,
                processing_time=time.time() - start_time,
                warnings=warnings,
                errors=errors,
            )

            return PartialResult(data=result, status=status, success_rate=success_rate)

        except Exception as e:
            context = ErrorContext(operation="file_read_corrupted", file_path=file_path)

            error = self.error_handler.handle_error(
                category=ErrorCategory.CORRUPTION,
                severity=ErrorSeverity.HIGH,
                message=f"Corrupted file read failed: {e}",
                context=context,
                exception=e,
            )

            errors.append(error.message)

            result = FileReadResult(
                content=None,
                file_info=file_info,
                encoding_used=None,
                success=False,
                partial_read=True,
                bytes_read=0,
                processing_time=time.time() - start_time,
                warnings=warnings,
                errors=errors,
            )

            return PartialResult(data=result, status=ResultStatus.FAILED, success_rate=0.0)

    def _stream_file_read(
        self, file_path: Path, file_info: FileInfo, encoding: Optional[str] = None
    ) -> PartialResult[FileReadResult]:
        """Stream read large file in chunks."""

        start_time = time.time()
        warnings = [f"Streaming large file ({file_info.size / 1024 / 1024:.1f}MB)"]
        errors = []

        try:
            # For very large files, we return a generator instead of loading all content
            def content_generator():
                target_encoding = encoding or file_info.encoding or "utf-8"

                if file_info.is_text:
                    with open(file_path, encoding=target_encoding, errors="replace") as f:
                        while True:
                            chunk = f.read(self.chunk_size)
                            if not chunk:
                                break
                            yield chunk
                else:
                    with open(file_path, "rb") as f:
                        while True:
                            chunk = f.read(self.chunk_size)
                            if not chunk:
                                break
                            yield chunk

            result = FileReadResult(
                content=content_generator(),  # Generator for streaming
                file_info=file_info,
                encoding_used=encoding or file_info.encoding,
                success=True,
                partial_read=True,  # Indicates streaming mode
                bytes_read=file_info.size,
                processing_time=time.time() - start_time,
                warnings=warnings,
                errors=errors,
            )

            self.stats["large_files_streamed"] += 1

            return PartialResult(
                data=result,
                status=ResultStatus.PARTIAL,
                success_rate=0.9,  # Slightly lower due to streaming complexity
            )

        except Exception as e:
            context = ErrorContext(
                operation="file_stream_read",
                file_path=file_path,
                additional_info={"file_size": file_info.size},
            )

            error = self.error_handler.handle_error(
                category=ErrorCategory.MEMORY,
                severity=ErrorSeverity.HIGH,
                message=f"Stream read failed: {e}",
                context=context,
                exception=e,
            )

            errors.append(error.message)

            result = FileReadResult(
                content=None,
                file_info=file_info,
                encoding_used=None,
                success=False,
                partial_read=False,
                bytes_read=0,
                processing_time=time.time() - start_time,
                warnings=warnings,
                errors=errors,
            )

            return PartialResult(data=result, status=ResultStatus.FAILED, success_rate=0.0)

    def _detect_file_type(self, file_path: Path) -> FileType:
        """Detect file type based on extension and content."""

        suffix = file_path.suffix.lower()

        type_map = {
            ".lean": FileType.LEAN,
            ".json": FileType.JSON,
            ".md": FileType.MARKDOWN,
            ".txt": FileType.TEXT,
            ".yaml": FileType.CONFIG,
            ".yml": FileType.CONFIG,
            ".toml": FileType.CONFIG,
            ".ini": FileType.CONFIG,
            ".cfg": FileType.CONFIG,
        }

        if suffix in type_map:
            return type_map[suffix]

        # Try content-based detection for files without clear extensions
        try:
            with open(file_path, "rb") as f:
                sample = f.read(1024)

            # Check if it's text-like
            try:
                sample.decode("utf-8")
                return FileType.TEXT
            except UnicodeDecodeError:
                return FileType.BINARY

        except Exception:
            return FileType.UNKNOWN

    def _detect_encoding(
        self, file_path: Path, sample_size: int = 10000
    ) -> Tuple[Optional[str], EncodingConfidence]:
        """Detect file encoding with confidence level."""

        try:
            with open(file_path, "rb") as f:
                sample = f.read(sample_size)

            if not sample:
                return "utf-8", EncodingConfidence.GUESS

            # Try chardet if available
            if HAS_CHARDET:
                detection = chardet.detect(sample)
                encoding = detection.get("encoding")
                confidence = detection.get("confidence", 0.0)

                if confidence > 0.9:
                    conf_level = EncodingConfidence.HIGH
                elif confidence > 0.7:
                    conf_level = EncodingConfidence.MEDIUM
                elif confidence > 0.5:
                    conf_level = EncodingConfidence.LOW
                else:
                    conf_level = EncodingConfidence.GUESS

                return encoding, conf_level

            # Fallback: try common encodings
            for encoding in self.encoding_fallbacks:
                try:
                    sample.decode(encoding)
                    return encoding, EncodingConfidence.GUESS
                except UnicodeDecodeError:
                    continue

            return None, EncodingConfidence.GUESS

        except Exception:
            return None, EncodingConfidence.GUESS

    def _read_text_file_with_fallback(self, file_path: Path, primary_encoding: str) -> str:
        """Read text file with encoding fallback."""

        encodings_to_try = [primary_encoding] + [
            enc for enc in self.encoding_fallbacks if enc != primary_encoding
        ]

        for encoding in encodings_to_try:
            try:
                with open(file_path, encoding=encoding) as f:
                    content = f.read()

                if encoding != primary_encoding:
                    self.stats["fallback_encodings_used"] += 1

                return content

            except UnicodeDecodeError:
                continue
            except Exception:
                break

        # Last resort: read with errors='replace'
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            raise

    def _decode_with_fallbacks(self, raw_content: bytes) -> str:
        """Decode bytes with fallback encodings."""

        for encoding in self.encoding_fallbacks:
            try:
                return raw_content.decode(encoding)
            except UnicodeDecodeError:
                continue

        # Last resort
        return raw_content.decode("utf-8", errors="replace")

    def _check_corruption(
        self, file_path: Path, file_type: FileType, encoding: Optional[str]
    ) -> bool:
        """Check for potential file corruption."""

        try:
            if file_type in [FileType.LEAN, FileType.TEXT, FileType.JSON]:
                # Check for binary data in text files
                with open(file_path, "rb") as f:
                    sample = f.read(1024)

                # Check for null bytes
                if b"\\x00" in sample:
                    return True

                # Check for encoding consistency
                if encoding:
                    try:
                        sample.decode(encoding)
                    except UnicodeDecodeError:
                        return True

            # For JSON files, try to parse a small sample
            if file_type == FileType.JSON:
                try:
                    pass

                    with open(file_path, encoding=encoding or "utf-8") as f:
                        # Try to parse first few characters to check basic structure
                        peek = f.read(100).strip()
                        if peek and not (
                            peek.startswith(("{", "[", '"'))
                            or peek[0].isdigit()
                            or peek.startswith(("true", "false", "null"))
                        ):
                            return True
                except Exception:
                    return True

            return False

        except Exception:
            return True  # If we can't check, assume corruption

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""

        hash_sha256 = hashlib.sha256()

        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)

            return hash_sha256.hexdigest()

        except Exception:
            return None

    def _estimate_processing_time(self, file_size: int, file_type: FileType) -> float:
        """Estimate processing time based on file size and type."""

        # Base processing rates (MB/second)
        rates = {
            FileType.LEAN: 5.0,  # Lean files need parsing
            FileType.TEXT: 10.0,  # Plain text is fast
            FileType.JSON: 3.0,  # JSON needs parsing
            FileType.BINARY: 50.0,  # Binary is fastest
            FileType.UNKNOWN: 5.0,
        }

        rate = rates.get(file_type, 5.0)
        size_mb = file_size / (1024 * 1024)

        return size_mb / rate

    def write_file_robust(
        self,
        file_path: Path,
        content: Union[str, bytes],
        encoding: str = "utf-8",
        backup: bool = True,
        atomic: bool = True,
    ) -> PartialResult[bool]:
        """Robustly write file with backup and atomic operations."""

        start_time = time.time()
        warnings = []

        try:
            # Create backup if requested and file exists
            backup_path = None
            if backup and file_path.exists():
                backup_path = file_path.with_suffix(f"{file_path.suffix}.backup_{int(time.time())}")
                shutil.copy2(file_path, backup_path)
                warnings.append(f"Created backup: {backup_path}")

            if atomic:
                # Atomic write using temporary file
                temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp_{os.getpid()}")

                try:
                    if isinstance(content, str):
                        with open(temp_path, "w", encoding=encoding) as f:
                            f.write(content)
                    else:
                        with open(temp_path, "wb") as f:
                            f.write(content)

                    # Atomic move
                    temp_path.replace(file_path)

                except Exception:
                    # Cleanup temp file on failure
                    if temp_path.exists():
                        temp_path.unlink()
                    raise
            else:
                # Direct write
                if isinstance(content, str):
                    with open(file_path, "w", encoding=encoding) as f:
                        f.write(content)
                else:
                    with open(file_path, "wb") as f:
                        f.write(content)

            return PartialResult(
                data=True,
                status=ResultStatus.COMPLETE,
                success_rate=1.0,
                metadata={
                    "processing_time": time.time() - start_time,
                    "backup_created": backup_path is not None,
                    "atomic_write": atomic,
                    "file_size": len(content) if content else 0,
                },
            )

        except Exception as e:
            context = ErrorContext(
                operation="file_write",
                file_path=file_path,
                additional_info={"atomic": atomic, "backup": backup},
            )

            error = self.error_handler.handle_error(
                category=ErrorCategory.FILE_ACCESS,
                severity=ErrorSeverity.HIGH,
                message=f"File write failed: {e}",
                context=context,
                exception=e,
            )

            return PartialResult(
                data=False,
                status=ResultStatus.FAILED,
                success_rate=0.0,
                errors=[error.message],
                metadata={"processing_time": time.time() - start_time},
            )

    def get_file_stats(self) -> Dict[str, Any]:
        """Get file handling statistics."""
        return {
            "statistics": self.stats,
            "cache_size": len(self.file_info_cache),
            "encoding_fallbacks": self.encoding_fallbacks,
            "max_file_size_mb": self.max_file_size_mb,
            "chunk_size": self.chunk_size,
        }

    def clear_cache(self):
        """Clear file info cache."""
        self.file_info_cache.clear()


# Utility functions
def is_safe_to_process(file_info: FileInfo, max_size_mb: int = 100) -> Tuple[bool, List[str]]:
    """Check if file is safe to process."""

    issues = []

    if not file_info.is_readable:
        issues.append("File is not readable")

    if file_info.potential_corruption:
        issues.append("Potential file corruption detected")

    if file_info.size > max_size_mb * 1024 * 1024:
        issues.append(f"File too large ({file_info.size / 1024 / 1024:.1f}MB > {max_size_mb}MB)")

    if file_info.encoding_confidence == EncodingConfidence.GUESS and file_info.is_text:
        issues.append("Uncertain text encoding")

    return len(issues) == 0, issues
