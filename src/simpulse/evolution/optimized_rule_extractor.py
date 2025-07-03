"""Optimized rule extractor with parallel processing and efficient caching."""

import concurrent.futures
import hashlib
import logging
import mmap
import pickle
import re
from functools import lru_cache
from pathlib import Path
from threading import Lock

from .models import ModuleRules, SimpDirection, SimpPriority, SimpRule, SourceLocation

logger = logging.getLogger(__name__)


class OptimizedRuleExtractor:
    """High-performance rule extractor with parallel processing and caching."""

    # Compiled regex patterns (class-level for reuse)
    SIMP_PATTERN = re.compile(
        r"@\[simp(?:(?:\s*,\s*priority\s*:=\s*\d+)?(?:\s+(?:high|low|\d+))?(?:\s*[↓←])?)?\]",
        re.MULTILINE,
    )

    DECLARATION_PATTERN = re.compile(
        r"(theorem|lemma|def|instance|axiom)\s+([a-zA-Z_][a-zA-Z0-9_\']*)", re.MULTILINE
    )

    MODULE_PATTERN = re.compile(r"namespace\s+([a-zA-Z_][a-zA-Z0-9_\.]*)")
    IMPORT_PATTERN = re.compile(r"^import\s+([a-zA-Z_][a-zA-Z0-9_\.]*)", re.MULTILINE)

    def __init__(self, cache_dir: Path = None, max_workers: int = None):
        """Initialize with optional persistent cache and worker configuration."""
        self.cache_dir = cache_dir or Path.home() / ".simpulse" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Memory cache with size limit
        self._memory_cache: dict[Path, ModuleRules] = {}
        self._cache_lock = Lock()
        self._max_cache_size = 1000  # Maximum files in memory

        # Worker pool configuration
        import os

        cpu_count = os.cpu_count() or 1
        self.max_workers = max_workers or min(8, cpu_count + 2)

        # Statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "files_processed": 0,
            "rules_extracted": 0,
        }

    def extract_rules_from_file(self, file_path: Path) -> ModuleRules:
        """Extract rules with caching and optimization."""
        # Check memory cache first
        with self._cache_lock:
            if file_path in self._memory_cache:
                self.stats["cache_hits"] += 1
                return self._memory_cache[file_path]

        # Check disk cache
        cached_result = self._load_from_disk_cache(file_path)
        if cached_result:
            self.stats["cache_hits"] += 1
            self._update_memory_cache(file_path, cached_result)
            return cached_result

        # Cache miss - extract rules
        self.stats["cache_misses"] += 1
        result = self._extract_rules_uncached(file_path)

        # Update caches
        self._save_to_disk_cache(file_path, result)
        self._update_memory_cache(file_path, result)

        return result

    def extract_rules_from_project(
        self, project_path: Path, exclude_patterns: list[str] = None
    ) -> dict[Path, ModuleRules]:
        """Extract rules from entire project in parallel."""
        exclude_patterns = exclude_patterns or ["lake-packages", ".lake", "build"]

        # Collect all Lean files
        lean_files = []
        for pattern in ["**/*.lean"]:
            for file_path in project_path.glob(pattern):
                # Skip excluded paths
                if not any(excl in str(file_path) for excl in exclude_patterns):
                    lean_files.append(file_path)

        if not lean_files:
            logger.warning(f"No Lean files found in {project_path}")
            return {}

        logger.info(f"Processing {len(lean_files)} files with {self.max_workers} workers")

        # Process files in parallel
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.extract_rules_from_file, file_path): file_path
                for file_path in lean_files
            }

            # Collect results with progress tracking
            for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
                file_path = future_to_file[future]
                try:
                    module_rules = future.result()
                    results[file_path] = module_rules

                    if (i + 1) % 10 == 0:
                        logger.debug(f"Processed {i + 1}/{len(lean_files)} files")

                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    results[file_path] = ModuleRules(
                        module_name=file_path.stem, file_path=file_path, rules=[]
                    )

        return results

    def _extract_rules_uncached(self, file_path: Path) -> ModuleRules:
        """Extract rules without caching - optimized implementation."""
        if not file_path.exists():
            return ModuleRules(module_name=file_path.stem, file_path=file_path, rules=[])

        try:
            # Use memory-mapped file for large files
            if file_path.stat().st_size > 1024 * 1024:  # 1MB threshold
                content = self._read_with_mmap(file_path)
            else:
                content = file_path.read_text(encoding="utf-8")

            # Extract all components in one pass
            module_name = self._extract_module_name_fast(content, file_path)
            imports = self._extract_imports_fast(content)
            rules = self._extract_simp_rules_fast(content, file_path)

            self.stats["files_processed"] += 1
            self.stats["rules_extracted"] += len(rules)

            return ModuleRules(
                module_name=module_name,
                file_path=file_path,
                rules=rules,
                imports=imports,
                metadata={
                    "total_rules": len(rules),
                    "file_size": len(content),
                    "line_count": content.count("\n") + 1,
                },
            )

        except Exception as e:
            logger.error(f"Failed to extract from {file_path}: {e}")
            return ModuleRules(module_name=file_path.stem, file_path=file_path, rules=[])

    def _read_with_mmap(self, file_path: Path) -> str:
        """Read large files efficiently with memory mapping."""
        with open(file_path, encoding="utf-8") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped:
                return mmapped.read().decode("utf-8")

    @lru_cache(maxsize=256)
    def _extract_module_name_fast(self, content: str, file_path: Path) -> str:
        """Extract module name with caching."""
        match = self.MODULE_PATTERN.search(content)
        return match.group(1) if match else file_path.stem

    def _extract_imports_fast(self, content: str) -> list[str]:
        """Extract imports efficiently."""
        return [match.group(1) for match in self.IMPORT_PATTERN.finditer(content)]

    def _extract_simp_rules_fast(self, content: str, file_path: Path) -> list[SimpRule]:
        """Extract simp rules with optimized parsing."""
        rules = []
        lines = content.split("\n")
        line_starts = self._compute_line_starts(content)

        # Find all simp attributes
        for match in self.SIMP_PATTERN.finditer(content):
            attr_pos = match.start()
            line_num = self._position_to_line(attr_pos, line_starts)

            # Parse attribute
            attr_text = match.group(0)
            priority, direction = self._parse_simp_attribute_fast(attr_text)

            # Find associated declaration
            decl_info = self._find_declaration_fast(lines, line_num, content, attr_pos)
            if decl_info:
                rule = self._create_rule_fast(
                    decl_info, priority, direction, file_path, line_num + 1
                )
                if rule:
                    rules.append(rule)

        return rules

    def _compute_line_starts(self, content: str) -> list[int]:
        """Compute byte positions of line starts for fast line lookup."""
        starts = [0]
        for i, char in enumerate(content):
            if char == "\n":
                starts.append(i + 1)
        return starts

    def _position_to_line(self, pos: int, line_starts: list[int]) -> int:
        """Convert byte position to line number using binary search."""
        left, right = 0, len(line_starts) - 1

        while left <= right:
            mid = (left + right) // 2
            if line_starts[mid] <= pos:
                if mid == len(line_starts) - 1 or line_starts[mid + 1] > pos:
                    return mid
                left = mid + 1
            else:
                right = mid - 1

        return 0

    @lru_cache(maxsize=1024)
    def _parse_simp_attribute_fast(
        self, attr_text: str
    ) -> tuple[int | SimpPriority, SimpDirection]:
        """Parse simp attribute with caching."""
        # Extract priority
        priority = SimpPriority.DEFAULT

        # Check for priority := syntax
        priority_match = re.search(r"priority\s*:=\s*(\d+)", attr_text)
        if priority_match:
            priority = int(priority_match.group(1))
        elif "high" in attr_text:
            priority = SimpPriority.HIGH
        elif "low" in attr_text:
            priority = SimpPriority.LOW
        else:
            # Check for numeric priority without :=
            match = re.search(r"@\[simp\s+(\d+)", attr_text)
            if match:
                priority = int(match.group(1))

        # Extract direction
        direction = (
            SimpDirection.BACKWARD
            if ("↓" in attr_text or "←" in attr_text)
            else SimpDirection.FORWARD
        )

        return priority, direction

    def _find_declaration_fast(
        self, lines: list[str], attr_line: int, content: str, attr_pos: int
    ) -> dict | None:
        """Find declaration following attribute - optimized."""
        # Look ahead for declaration
        search_text = content[attr_pos : attr_pos + 500]  # Limited lookahead

        match = self.DECLARATION_PATTERN.search(search_text)
        if match:
            decl_type = match.group(1)
            decl_name = match.group(2)

            # Extract full declaration efficiently
            decl_start = attr_pos + match.start()
            decl_end = self._find_declaration_end(content, decl_start)
            declaration = content[decl_start:decl_end]

            return {
                "type": decl_type,
                "name": decl_name,
                "declaration": declaration,
                "line": attr_line + search_text[: match.start()].count("\n"),
            }

        return None

    def _find_declaration_end(self, content: str, start: int) -> int:
        """Find end of declaration using heuristics."""
        # Simple heuristic: declaration ends at next top-level construct
        # or double newline
        pos = start
        brace_count = 0
        paren_count = 0

        while pos < len(content):
            char = content[pos]

            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
            elif char == "(":
                paren_count += 1
            elif char == ")":
                paren_count -= 1
            elif char == "\n" and brace_count == 0 and paren_count == 0:
                # Check for double newline or next declaration
                if pos + 1 < len(content) and content[pos + 1] == "\n":
                    return pos

                # Check for next declaration
                next_chunk = content[pos : pos + 20]
                if any(kw in next_chunk for kw in ["theorem", "lemma", "def", "@["]):
                    return pos

            pos += 1

        return len(content)

    def _create_rule_fast(
        self,
        decl_info: dict,
        priority: int | SimpPriority,
        direction: SimpDirection,
        file_path: Path,
        line_num: int,
    ) -> SimpRule | None:
        """Create rule object - optimized."""
        try:
            location = SourceLocation(
                file=file_path, line=line_num, column=0, module=file_path.stem
            )

            return SimpRule(
                name=decl_info["name"],
                declaration=decl_info["declaration"],
                priority=priority,
                direction=direction,
                location=location,
                conditions=[],  # Skip complex parsing for speed
                pattern=None,  # Skip pattern extraction for speed
                rhs=None,  # Skip RHS extraction for speed
                metadata={"declaration_type": decl_info["type"], "file_path": str(file_path)},
            )
        except Exception:
            return None

    def _update_memory_cache(self, file_path: Path, result: ModuleRules):
        """Update memory cache with LRU eviction."""
        with self._cache_lock:
            # Evict oldest entries if cache is full
            if len(self._memory_cache) >= self._max_cache_size:
                # Simple FIFO eviction (could be improved to LRU)
                oldest = next(iter(self._memory_cache))
                del self._memory_cache[oldest]

            self._memory_cache[file_path] = result

    def _get_cache_key(self, file_path: Path) -> str:
        """Generate cache key based on file path and modification time."""
        stat = file_path.stat()
        key_data = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _load_from_disk_cache(self, file_path: Path) -> ModuleRules | None:
        """Load from persistent disk cache."""
        cache_key = self._get_cache_key(file_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.debug(f"Cache load failed for {file_path}: {e}")
                cache_file.unlink()  # Remove corrupted cache

        return None

    def _save_to_disk_cache(self, file_path: Path, result: ModuleRules):
        """Save to persistent disk cache."""
        cache_key = self._get_cache_key(file_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.debug(f"Cache save failed for {file_path}: {e}")

    def clear_cache(self):
        """Clear all caches."""
        with self._cache_lock:
            self._memory_cache.clear()

        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception:
                pass

        # Reset stats
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "files_processed": 0,
            "rules_extracted": 0,
        }

    def get_statistics(self) -> dict:
        """Get extraction statistics."""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = self.stats["cache_hits"] / total_requests if total_requests > 0 else 0

        return {
            **self.stats,
            "cache_hit_rate": hit_rate,
            "memory_cache_size": len(self._memory_cache),
            "disk_cache_files": len(list(self.cache_dir.glob("*.pkl"))),
        }
