"""Smart pattern-based optimizer using sophisticated analysis."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from pydantic import Field

from ..analyzer import SimpRule
from ..errors import ErrorHandler
from .optimizer import OptimizationChange, OptimizationResult
from .pattern_analyzer import ContextPattern, PatternAnalyzer, RulePattern


class SmartOptimizationResult(OptimizationResult):
    """Enhanced optimization result with pattern insights."""

    optimization_insights: List[str] = Field(default_factory=list)
    pattern_data: Dict[str, Any] = Field(default_factory=dict)


class SmartPatternOptimizer:
    """Optimizer using sophisticated pattern analysis for better results."""

    def __init__(self):
        self.pattern_analyzer = PatternAnalyzer()
        self.error_handler = ErrorHandler(logging.getLogger(__name__))
        from ..evolution.rule_extractor import RuleExtractor

        self.extractor = RuleExtractor()

    def analyze(self, project_path: Path):
        """Enhanced analysis with pattern detection."""
        # Extract all rules
        all_rules = []
        failed_files = []

        lean_files = list(project_path.glob("**/*.lean"))
        for lean_file in lean_files:
            if "lake-packages" not in str(lean_file):
                try:
                    module_rules = self.extractor.extract_rules_from_file(lean_file)
                    all_rules.extend(module_rules.rules)
                except Exception:
                    failed_files.append(lean_file)

        # Perform pattern analysis
        print("🔍 Analyzing simp behavior patterns...")
        pattern_result = self.pattern_analyzer.analyze_patterns(project_path)

        # Build analysis result
        analysis = {
            "project_path": project_path,
            "rules": all_rules,
            "analysis_stats": {
                "total_files": len(lean_files),
                "successful_files": len(lean_files) - len(failed_files),
                "failed_files": len(failed_files),
                "total_rules": len(all_rules),
            },
            "pattern_analysis": pattern_result,
            "pattern_insights": pattern_result.optimization_insights,
        }

        return analysis

    def optimize(self, analysis) -> SmartOptimizationResult:
        """Generate optimizations using pattern analysis."""
        project_path = analysis["project_path"]
        rules = analysis["rules"]
        pattern_analysis = analysis.get("pattern_analysis")

        if not pattern_analysis:
            # Fallback to basic optimization
            return super().optimize(analysis)

        # Generate smart optimizations
        changes = self._optimize_with_patterns(
            rules,
            pattern_analysis.rule_patterns,
            pattern_analysis.context_patterns,
            pattern_analysis.rule_clusters,
        )

        # Calculate sophisticated improvement estimate
        estimated_improvement = self._estimate_pattern_based_improvement(
            changes, pattern_analysis.rule_patterns
        )

        return SmartOptimizationResult(
            project_path=project_path,
            rules_changed=len(changes),
            estimated_improvement=estimated_improvement,
            changes=changes,
            optimization_insights=pattern_analysis.optimization_insights[:5],
            pattern_data={
                "total_patterns_analyzed": len(pattern_analysis.rule_patterns),
                "contexts_identified": list(pattern_analysis.context_patterns.keys()),
                "rule_clusters": len(pattern_analysis.rule_clusters),
            },
        )

    def _optimize_with_patterns(
        self,
        rules: List[SimpRule],
        rule_patterns: Dict[str, RulePattern],
        context_patterns: Dict[str, ContextPattern],
        rule_clusters: List[Set[str]],
    ) -> List[OptimizationChange]:
        """Generate optimizations based on pattern analysis."""
        changes = []
        used_priorities = set()

        # Phase 1: Optimize high-performance rules (priority 50-200)
        high_performers = self._get_high_performance_rules(rule_patterns)
        priority = 50
        for rule_name, pattern in high_performers[:10]:
            rule = self._find_rule_by_name(rules, rule_name)
            if rule and rule.priority is None:
                changes.append(
                    self._create_change(
                        rule,
                        priority,
                        f"High performer: {pattern.success_rate:.0%} success, "
                        f"{pattern.total_attempts} attempts, "
                        f"{pattern.avg_application_time:.1f}ms avg time",
                    )
                )
                used_priorities.add(priority)
                priority += 15

        # Phase 2: Optimize context-specific champions (priority 200-400)
        priority = 200
        for context_type, ctx_pattern in context_patterns.items():
            context_champions = sorted(
                ctx_pattern.rule_performance.items(), key=lambda x: x[1], reverse=True
            )[:5]

            for rule_name, success_rate in context_champions:
                if rule_name not in [c.rule_name for c in changes]:
                    rule = self._find_rule_by_name(rules, rule_name)
                    if rule and rule.priority is None:
                        changes.append(
                            self._create_change(
                                rule,
                                priority,
                                f"{context_type.title()} specialist: "
                                f"{success_rate:.0%} success in context",
                            )
                        )
                        used_priorities.add(priority)
                        priority += 20

        # Phase 3: Optimize rule clusters (priority 400-600)
        priority = 400
        for cluster in rule_clusters[:5]:  # Top 5 clusters
            cluster_priority = priority
            for rule_name in sorted(cluster)[:4]:  # Up to 4 rules per cluster
                if rule_name not in [c.rule_name for c in changes]:
                    rule = self._find_rule_by_name(rules, rule_name)
                    if rule and rule.priority is None:
                        pattern = rule_patterns.get(rule_name)
                        if pattern:
                            changes.append(
                                self._create_change(
                                    rule,
                                    cluster_priority,
                                    f"Cluster member: works well with "
                                    f"{len(pattern.co_occurring_rules)} other rules",
                                )
                            )
                            used_priorities.add(cluster_priority)
                            cluster_priority += 5
            priority += 50

        # Phase 4: Optimize fast & effective rules (priority 600-800)
        priority = 600
        fast_effective = self._get_fast_effective_rules(rule_patterns)
        for rule_name, pattern in fast_effective[:10]:
            if rule_name not in [c.rule_name for c in changes]:
                rule = self._find_rule_by_name(rules, rule_name)
                if rule and rule.priority is None:
                    changes.append(
                        self._create_change(
                            rule,
                            priority,
                            f"Fast & effective: {pattern.avg_application_time:.1f}ms, "
                            f"{pattern.success_rate:.0%} success",
                        )
                    )
                    used_priorities.add(priority)
                    priority += 20

        # Phase 5: Deprioritize problematic rules (priority 1500-2000)
        priority = 1500
        problematic = self._get_problematic_rules(rule_patterns)
        for rule_name, pattern in problematic[:10]:
            rule = self._find_rule_by_name(rules, rule_name)
            if rule and rule.priority is None:
                changes.append(
                    self._create_change(
                        rule,
                        priority,
                        f"Deprioritized: {pattern.success_rate:.0%} success, "
                        f"{pattern.avg_search_depth:.1f} avg depth",
                    )
                )
                used_priorities.add(priority)
                priority += 50

        return changes

    def _get_high_performance_rules(
        self, rule_patterns: Dict[str, RulePattern]
    ) -> List[Tuple[str, RulePattern]]:
        """Get rules with best overall performance."""
        scored_rules = []

        for rule_name, pattern in rule_patterns.items():
            if pattern.total_attempts >= 10:  # Minimum attempts threshold
                # Composite score based on multiple factors
                score = (
                    pattern.success_rate * 0.4  # 40% weight on success
                    + (1.0 / max(pattern.avg_search_depth, 1.0)) * 0.2  # 20% on shallow search
                    + (1.0 / max(pattern.avg_application_time, 0.1)) * 0.2  # 20% on speed
                    + min(pattern.total_attempts / 100, 1.0) * 0.2  # 20% on frequency
                )
                scored_rules.append((rule_name, pattern, score))

        # Sort by score
        scored_rules.sort(key=lambda x: x[2], reverse=True)
        return [(name, pattern) for name, pattern, _ in scored_rules]

    def _get_fast_effective_rules(
        self, rule_patterns: Dict[str, RulePattern]
    ) -> List[Tuple[str, RulePattern]]:
        """Get rules that are both fast and effective."""
        fast_effective = []

        for rule_name, pattern in rule_patterns.items():
            if (
                pattern.avg_application_time < 1.0  # Fast (< 1ms)
                and pattern.success_rate > 0.6  # Effective (> 60%)
                and pattern.total_attempts >= 5
            ):  # Minimum usage
                fast_effective.append((rule_name, pattern))

        # Sort by combined metric
        fast_effective.sort(
            key=lambda x: x[1].success_rate / max(x[1].avg_application_time, 0.1), reverse=True
        )
        return fast_effective

    def _get_problematic_rules(
        self, rule_patterns: Dict[str, RulePattern]
    ) -> List[Tuple[str, RulePattern]]:
        """Get rules that should be deprioritized."""
        problematic = []

        for rule_name, pattern in rule_patterns.items():
            # Rules with poor characteristics
            if (
                pattern.success_rate < 0.3  # Low success
                or pattern.avg_search_depth > 4.0  # Deep search required
                or pattern.avg_application_time > 5.0
            ):  # Slow
                problematic.append((rule_name, pattern))

        # Sort by how problematic they are (worst first)
        problematic.sort(
            key=lambda x: x[1].success_rate - x[1].avg_search_depth * 0.1,
        )
        return problematic

    def _find_rule_by_name(self, rules: List[SimpRule], name: str) -> SimpRule:
        """Find a rule by name."""
        for rule in rules:
            if rule.name == name:
                return rule
        return None

    def _create_change(self, rule: SimpRule, priority: int, reason: str) -> OptimizationChange:
        """Create an optimization change."""
        return OptimizationChange(
            rule_name=rule.name,
            file_path=str(rule.file_path),
            old_priority=rule.priority if rule.priority is not None else 1000,
            new_priority=priority,
            reason=reason,
        )

    def _estimate_pattern_based_improvement(
        self, changes: List[OptimizationChange], rule_patterns: Dict[str, RulePattern]
    ) -> int:
        """Estimate improvement based on pattern analysis."""
        if not changes:
            return 0

        total_impact = 0.0

        for change in changes:
            pattern = rule_patterns.get(change.rule_name)
            if pattern:
                # Calculate impact based on rule characteristics
                frequency_impact = min(pattern.total_attempts / 100, 1.0) * 0.3
                success_impact = pattern.success_rate * 0.3

                # Priority change impact
                priority_delta = abs(change.old_priority - change.new_priority) / 1000
                priority_impact = min(priority_delta, 1.0) * 0.4

                rule_impact = frequency_impact + success_impact + priority_impact
                total_impact += rule_impact

        # Convert to percentage (max 75% improvement)
        estimated_improvement = min(int(total_impact * 10), 75)

        return estimated_improvement

    def apply(
        self, optimization: SmartOptimizationResult, project_path: Path, create_backup: bool = True
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


def demonstrate_smart_optimization(project_path: Path):
    """Demonstrate the smart pattern-based optimization."""
    print("=" * 70)
    print("🧠 Smart Pattern-Based Simp Optimization Demo")
    print("=" * 70)

    # Create optimizer
    optimizer = SmartPatternOptimizer()

    # Analyze project
    print(f"\n📊 Analyzing project: {project_path}")
    analysis = optimizer.analyze(project_path)

    print(f"\n📈 Pattern Analysis Results:")
    print(f"   • Total rules found: {len(analysis['rules'])}")
    print(f"   • Pattern insights: {len(analysis.get('pattern_insights', []))}")

    if analysis.get("pattern_insights"):
        print(f"\n💡 Key Insights:")
        for i, insight in enumerate(analysis["pattern_insights"][:3], 1):
            print(f"   {i}. {insight}")

    # Generate optimizations
    print(f"\n⚡ Generating smart optimizations...")
    result = optimizer.optimize(analysis)

    print(f"\n🎯 Optimization Results:")
    print(f"   • Rules optimized: {result.rules_changed}")
    print(f"   • Estimated improvement: {result.estimated_improvement}%")

    if result.pattern_data:
        print(f"\n📊 Pattern Data:")
        print(f"   • Patterns analyzed: {result.pattern_data['total_patterns_analyzed']}")
        print(f"   • Contexts identified: {', '.join(result.pattern_data['contexts_identified'])}")
        print(f"   • Rule clusters found: {result.pattern_data['rule_clusters']}")

    # Show sample optimizations
    if result.changes:
        print(f"\n🔧 Sample Optimizations:")
        for change in result.changes[:5]:
            print(f"\n   Rule: {change.rule_name}")
            print(f"   Priority: {change.old_priority} → {change.new_priority}")
            print(f"   Reason: {change.reason}")

    return result
