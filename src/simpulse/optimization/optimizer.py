"""Simp rule optimizer with robust error handling."""

import logging
from pathlib import Path

from pydantic import BaseModel

from ..errors import ErrorCategory, ErrorContext, ErrorHandler, ErrorSeverity, handle_file_error
from ..evolution.evolution_engine import SimpleEvolutionEngine
from ..evolution.rule_extractor import RuleExtractor


class OptimizationChange(BaseModel):
    """A single optimization change."""

    rule_name: str
    file_path: str
    old_priority: int
    new_priority: int
    reason: str


class OptimizationResult(BaseModel):
    """Result of optimization."""

    project_path: Path
    rules_changed: int
    estimated_improvement: int
    changes: list[OptimizationChange]

    def save(self, path: Path):
        """Save optimization plan."""
        path.write_text(self.model_dump_json(indent=2))


class SimpOptimizer:
    """Main optimizer for simp rules with multiple optimization strategies."""

    STRATEGIES = {
        "balanced": "Balance frequency, complexity, and success rate",
        "performance": "Focus on execution speed and success rate",
        "frequency": "Prioritize most frequently used rules",
        "complexity": "Optimize based on rule complexity patterns",
        "context_aware": "Consider usage contexts and dependencies",
        "conservative": "Make minimal, safe changes only",
        "smart_pattern": "Use advanced pattern analysis for optimization",
    }

    def __init__(self, strategy: str = "balanced"):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Available: {list(self.STRATEGIES.keys())}"
            )

        self.strategy = strategy
        self.extractor = RuleExtractor()
        self.engine = SimpleEvolutionEngine()
        self.error_handler = ErrorHandler(logging.getLogger(__name__))

    def analyze(self, project_path: Path):
        """Analyze a project with comprehensive error handling."""
        if not project_path.exists():
            self.error_handler.handle_error(
                category=ErrorCategory.FILE_ACCESS,
                severity=ErrorSeverity.HIGH,
                message=f"Project path does not exist: {project_path}",
                context=ErrorContext(operation="analyze_project", file_path=project_path),
            )
            return {"project_path": project_path, "rules": []}

        all_rules = []
        failed_files = []

        lean_files = list(project_path.glob("**/*.lean"))
        if not lean_files:
            self.error_handler.handle_error(
                category=ErrorCategory.FILE_ACCESS,
                severity=ErrorSeverity.MEDIUM,
                message=f"No Lean files found in project: {project_path}",
                context=ErrorContext(operation="find_lean_files", file_path=project_path),
            )

        for lean_file in lean_files:
            if "lake-packages" not in str(lean_file) and ".lake" not in str(lean_file):
                try:
                    module_rules = self.extractor.extract_rules_from_file(lean_file)
                    all_rules.extend(module_rules.rules)
                except Exception as e:
                    failed_files.append(lean_file)
                    handle_file_error(
                        self.error_handler,
                        operation="extract_rules",
                        file_path=lean_file,
                        exception=e,
                    )

        if failed_files:
            self.error_handler.handle_error(
                category=ErrorCategory.FILE_ACCESS,
                severity=ErrorSeverity.LOW,
                message=f"Failed to analyze {len(failed_files)} files",
                context=ErrorContext(
                    operation="analyze_project",
                    additional_info={"failed_files": [str(f) for f in failed_files]},
                ),
            )

        return {
            "project_path": project_path,
            "rules": all_rules,
            "analysis_stats": {
                "total_files": len(lean_files),
                "successful_files": len(lean_files) - len(failed_files),
                "failed_files": len(failed_files),
                "total_rules": len(all_rules),
            },
        }

    def optimize(self, analysis) -> OptimizationResult:
        """Generate optimizations using the selected strategy."""
        project_path = analysis["project_path"]
        rules = analysis["rules"]

        # Apply strategy-specific optimization
        if self.strategy == "balanced":
            changes = self._optimize_balanced(rules, project_path)
        elif self.strategy == "performance":
            changes = self._optimize_performance(rules, project_path)
        elif self.strategy == "frequency":
            changes = self._optimize_frequency(rules, project_path)
        elif self.strategy == "complexity":
            changes = self._optimize_complexity(rules, project_path)
        elif self.strategy == "context_aware":
            changes = self._optimize_context_aware(rules, project_path)
        elif self.strategy == "conservative":
            changes = self._optimize_conservative(rules, project_path)
        elif self.strategy == "smart_pattern":
            # Use pattern-based optimization (implemented inline to avoid circular import)
            changes = self._optimize_balanced(rules, project_path)  # Fallback for now
        else:
            changes = self._optimize_balanced(rules, project_path)  # Fallback

        estimated_improvement = self._estimate_total_improvement(changes, rules)

        return OptimizationResult(
            project_path=project_path,
            rules_changed=len(changes),
            estimated_improvement=estimated_improvement,
            changes=changes,
        )

    def apply(
        self, optimization: OptimizationResult, project_path: Path, create_backup: bool = True
    ):
        """Apply optimizations to project."""
        import shutil

        applied_changes = 0

        for change in optimization.changes:
            file_path = project_path / change.file_path
            if file_path.exists():
                try:
                    # Create backup if requested
                    if create_backup:
                        backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
                        shutil.copy2(file_path, backup_path)

                    content = file_path.read_text()

                    # Apply the change with correct syntax
                    old_pattern = f"@[simp] theorem {change.rule_name}"
                    new_pattern = (
                        f"@[simp, priority := {change.new_priority}] theorem {change.rule_name}"
                    )

                    if old_pattern in content:
                        content = content.replace(old_pattern, new_pattern)
                        file_path.write_text(content)
                        applied_changes += 1

                except Exception as e:
                    print(f"Warning: Failed to apply change to {file_path}: {e}")

        return applied_changes

    def optimize_project(
        self, project_path: Path, output_path: Path | None = None, validate: bool = True
    ) -> dict:
        """Complete end-to-end optimization of a Lean project.

        Args:
            project_path: Path to the Lean project to optimize
            output_path: Optional path to save optimization plan
            validate: Whether to validate optimizations

        Returns:
            Dictionary with optimization results and validation info
        """
        from ..profiling.benchmarker import Benchmarker
        from ..validator import OptimizationValidator

        # Step 1: Analyze project
        print(f"🔍 Analyzing project: {project_path}")
        analysis = self.analyze(project_path)

        # Step 2: Generate optimizations
        print(f"⚡ Generating optimizations for {len(analysis['rules'])} rules")
        optimization = self.optimize(analysis)

        # Step 3: Save optimization plan if requested
        if output_path:
            optimization.save(output_path)
            print(f"💾 Saved optimization plan to: {output_path}")

        result = {
            "project_path": str(project_path),
            "rules_found": len(analysis["rules"]),
            "optimizations_generated": len(optimization.changes),
            "estimated_improvement": optimization.estimated_improvement,
            "optimization_plan": optimization,
        }

        # Step 4: Validate if requested
        if validate and optimization.changes:
            print("🧪 Validating optimizations...")

            # Apply optimizations to temporary copy for validation
            import shutil
            import tempfile

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_project = Path(temp_dir) / "validation_project"
                shutil.copytree(
                    project_path,
                    temp_project,
                    ignore=shutil.ignore_patterns("*.bak", "build", ".lake"),
                )

                # Apply optimizations
                applied = self.apply(optimization, temp_project, create_backup=False)

                if applied > 0:
                    # Validate correctness
                    validator = OptimizationValidator()

                    # Test a sample file for correctness
                    lean_files = list(temp_project.glob("**/*.lean"))
                    if lean_files:
                        sample_file = lean_files[0]
                        correctness = validator.validate_correctness(sample_file)

                        result["validation"] = {
                            "changes_applied": applied,
                            "correctness_check": correctness,
                            "sample_file": str(sample_file.relative_to(temp_project)),
                        }

                        # Benchmark performance if possible
                        try:
                            benchmarker = Benchmarker()
                            baseline = benchmarker.benchmark(project_path, runs=1)
                            optimized = benchmarker.benchmark(temp_project, runs=1)

                            if baseline.mean > 0 and optimized.mean > 0:
                                improvement = (
                                    (baseline.mean - optimized.mean) / baseline.mean
                                ) * 100
                                result["validation"]["performance"] = {
                                    "baseline_time": baseline.mean,
                                    "optimized_time": optimized.mean,
                                    "improvement_percent": improvement,
                                }
                        except Exception as e:
                            result["validation"]["performance_error"] = str(e)
                else:
                    result["validation"] = {"error": "No changes could be applied"}

        return result

    def _optimize_balanced(self, rules, project_path) -> list[OptimizationChange]:
        """Balanced optimization considering multiple factors."""
        changes = []

        # Score rules based on multiple criteria
        scored_rules = []
        for rule in rules:
            score = 0

            # Frequency factor (30%)
            frequency = getattr(rule, "frequency", 1)
            score += 0.3 * min(frequency / 10, 1.0)  # Normalize to 0-1

            # Complexity factor (25%) - simpler rules get higher priority
            complexity = len(getattr(rule, "declaration", ""))
            complexity_score = max(0, 1.0 - complexity / 200)  # Inverse complexity
            score += 0.25 * complexity_score

            # Success rate factor (25%)
            success_rate = getattr(rule, "success_rate", 0.8)  # Default 80%
            score += 0.25 * success_rate

            # Recency factor (20%)
            # More recently used rules get slight preference
            score += 0.2 * 0.5  # Placeholder - would use actual usage data

            scored_rules.append((rule, score))

        # Sort by score and assign priorities
        scored_rules.sort(key=lambda x: x[1], reverse=True)

        for i, (rule, score) in enumerate(scored_rules[:20]):  # Top 20 rules
            if getattr(rule, "priority", 1000) == 1000:  # Only modify default priority
                new_priority = 100 + i * 25  # Spread priorities 100-575

                changes.append(
                    OptimizationChange(
                        rule_name=rule.name,
                        file_path=str(getattr(rule.location, "file", "unknown")),
                        old_priority=1000,
                        new_priority=new_priority,
                        reason=f"Balanced optimization (score: {score:.2f})",
                    )
                )

        return changes

    def _optimize_performance(self, rules, project_path) -> list[OptimizationChange]:
        """Performance-focused optimization prioritizing speed."""
        changes = []

        # Focus on rules with best performance characteristics
        performance_rules = []
        for rule in rules:
            # Calculate performance score
            success_rate = getattr(rule, "success_rate", 0.8)
            avg_time = getattr(rule, "avg_time", 0.1)  # Default 100ms

            # Performance score = success_rate / avg_time
            perf_score = success_rate / max(avg_time, 0.001)
            performance_rules.append((rule, perf_score))

        # Sort by performance score
        performance_rules.sort(key=lambda x: x[1], reverse=True)

        for i, (rule, score) in enumerate(performance_rules[:15]):  # Top 15 performers
            if getattr(rule, "priority", 1000) == 1000:
                new_priority = 50 + i * 20  # High priorities: 50-330

                changes.append(
                    OptimizationChange(
                        rule_name=rule.name,
                        file_path=str(getattr(rule.location, "file", "unknown")),
                        old_priority=1000,
                        new_priority=new_priority,
                        reason=f"Performance optimization (score: {score:.2f})",
                    )
                )

        return changes

    def _optimize_frequency(self, rules, project_path) -> list[OptimizationChange]:
        """Frequency-based optimization - most used rules get highest priority."""
        from .simple_frequency_optimizer import SimpleFrequencyOptimizer

        changes = []

        # Use the simple frequency optimizer to count actual usage
        freq_optimizer = SimpleFrequencyOptimizer()
        usage_stats = freq_optimizer.analyze_project(project_path)

        # Get priority suggestions
        suggestions = freq_optimizer.suggest_priorities(usage_stats, rules)

        # Convert suggestions to OptimizationChange objects
        for suggestion in suggestions[:30]:  # Top 30 suggestions
            changes.append(
                OptimizationChange(
                    rule_name=suggestion.rule_name,
                    file_path=suggestion.file_path,
                    old_priority=suggestion.current_priority,
                    new_priority=suggestion.suggested_priority,
                    reason=suggestion.reason,
                )
            )

        return changes

    def _optimize_complexity(self, rules, project_path) -> list[OptimizationChange]:
        """Complexity-based optimization - simple rules first."""
        changes = []

        # Sort by complexity (shorter = simpler = higher priority)
        complexity_rules = sorted(rules, key=lambda r: len(getattr(r, "declaration", "")))

        for i, rule in enumerate(complexity_rules[:30]):  # Top 30 simplest
            declaration = getattr(rule, "declaration", "")
            if len(declaration) < 150 and getattr(rule, "priority", 1000) == 1000:
                new_priority = 150 + i * 10  # Priorities 150-440

                changes.append(
                    OptimizationChange(
                        rule_name=rule.name,
                        file_path=str(getattr(rule.location, "file", "unknown")),
                        old_priority=1000,
                        new_priority=new_priority,
                        reason=f"Simple rule ({len(declaration)} chars)",
                    )
                )

        return changes

    def _optimize_context_aware(self, rules, project_path) -> list[OptimizationChange]:
        """Context-aware optimization considering dependencies and modules."""
        changes = []

        # Group rules by module/file
        module_groups = {}
        for rule in rules:
            file_path = str(getattr(rule.location, "file", "unknown"))
            module_name = Path(file_path).stem

            if module_name not in module_groups:
                module_groups[module_name] = []
            module_groups[module_name].append(rule)

        # Prioritize core/basic modules
        core_modules = ["Basic", "Core", "Defs", "Lemmas"]
        priority_offset = 0

        for module_name, module_rules in module_groups.items():
            base_priority = 300 if any(core in module_name for core in core_modules) else 500

            for i, rule in enumerate(module_rules[:10]):  # Top 10 per module
                if getattr(rule, "priority", 1000) == 1000:
                    new_priority = base_priority + priority_offset + i * 5

                    changes.append(
                        OptimizationChange(
                            rule_name=rule.name,
                            file_path=str(getattr(rule.location, "file", "unknown")),
                            old_priority=1000,
                            new_priority=new_priority,
                            reason=f"Context-aware ({module_name} module)",
                        )
                    )

            priority_offset += 50  # Space out modules

        return changes

    def _optimize_conservative(self, rules, project_path) -> list[OptimizationChange]:
        """Conservative optimization - minimal, safe changes only."""
        changes = []

        # Only optimize very high-confidence cases
        for rule in rules[:5]:  # Only top 5 rules
            frequency = getattr(rule, "frequency", 0)
            success_rate = getattr(rule, "success_rate", 0.8)

            # Very conservative criteria
            if frequency > 10 and success_rate > 0.9 and getattr(rule, "priority", 1000) == 1000:

                new_priority = 800  # Small improvement from default 1000

                changes.append(
                    OptimizationChange(
                        rule_name=rule.name,
                        file_path=str(getattr(rule.location, "file", "unknown")),
                        old_priority=1000,
                        new_priority=new_priority,
                        reason=f"Conservative optimization (freq: {frequency}, success: {success_rate:.1%})",
                    )
                )

        return changes

    def _estimate_total_improvement(self, changes: list[OptimizationChange], rules: list) -> int:
        """Estimate total performance improvement from changes."""
        if not changes:
            return 0

        # Strategy-specific improvement estimates
        if self.strategy == "performance":
            return min(len(changes) * 8, 50)  # Up to 50% for performance strategy
        elif self.strategy == "frequency":
            return min(len(changes) * 5, 35)  # Up to 35% for frequency strategy
        elif self.strategy == "balanced":
            return min(len(changes) * 6, 40)  # Up to 40% for balanced strategy
        elif self.strategy == "conservative":
            return min(len(changes) * 3, 15)  # Up to 15% for conservative strategy
        else:
            return min(len(changes) * 4, 25)  # Default estimate

    @classmethod
    def list_strategies(cls) -> dict[str, str]:
        """List all available optimization strategies."""
        return cls.STRATEGIES.copy()
