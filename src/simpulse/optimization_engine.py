"""
Evidence-Based Optimization Engine for Lean 4 Simp Rules

Generates optimization recommendations based on real diagnostic data from Lean 4.8.0+,
replacing theoretical estimates with evidence-based analysis.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .diagnostic_parser import DiagnosticAnalysis, SimpTheoremUsage

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimizations that can be applied."""

    PRIORITY_INCREASE = "priority_increase"  # Increase priority for frequently used theorems
    PRIORITY_DECREASE = "priority_decrease"  # Decrease priority for rarely used theorems
    REMOVE_INEFFICIENT = "remove_inefficient"  # Remove simp attribute from inefficient theorems
    FIX_LOOP = "fix_loop"  # Fix looping simp theorems
    REORDER_LEMMAS = "reorder_lemmas"  # Reorder lemma sets for better efficiency


@dataclass
class OptimizationRecommendation:
    """A single optimization recommendation based on diagnostic evidence."""

    theorem_name: str
    file_path: Path
    line_number: int
    optimization_type: OptimizationType
    current_priority: int
    recommended_priority: int | None
    evidence_score: float  # 0-100, higher means stronger evidence
    reason: str
    expected_impact: str  # Description of expected performance impact

    @property
    def priority_change(self) -> int:
        """Calculate priority change (negative means higher priority)."""
        if self.recommended_priority is None:
            return 0
        return self.recommended_priority - self.current_priority


@dataclass
class OptimizationPlan:
    """Complete optimization plan with evidence-based recommendations."""

    recommendations: list[OptimizationRecommendation] = field(default_factory=list)
    high_confidence: list[OptimizationRecommendation] = field(default_factory=list)
    medium_confidence: list[OptimizationRecommendation] = field(default_factory=list)
    low_confidence: list[OptimizationRecommendation] = field(default_factory=list)

    def __post_init__(self):
        # Categorize recommendations by confidence level
        for rec in self.recommendations:
            if rec.evidence_score >= 80:
                self.high_confidence.append(rec)
            elif rec.evidence_score >= 50:
                self.medium_confidence.append(rec)
            else:
                self.low_confidence.append(rec)

    @property
    def total_recommendations(self) -> int:
        return len(self.recommendations)

    def get_by_type(self, opt_type: OptimizationType) -> list[OptimizationRecommendation]:
        """Get recommendations by optimization type."""
        return [rec for rec in self.recommendations if rec.optimization_type == opt_type]


class OptimizationEngine:
    """Generates evidence-based optimization recommendations."""

    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.simp_rules_cache: dict[str, tuple[Path, int, int]] = (
            {}
        )  # name -> (path, line, priority)

    def analyze_and_recommend(self, diagnostic_analysis: DiagnosticAnalysis) -> OptimizationPlan:
        """Generate optimization recommendations based on diagnostic analysis."""
        logger.info("Analyzing diagnostic data for optimization opportunities...")

        # Scan for simp rules in the codebase
        self._scan_simp_rules()

        recommendations = []

        # 1. Recommend priority increases for highly used theorems
        recommendations.extend(self._recommend_priority_increases(diagnostic_analysis))

        # 2. Recommend priority decreases for inefficient theorems
        recommendations.extend(self._recommend_priority_decreases(diagnostic_analysis))

        # 3. Recommend fixes for looping theorems
        recommendations.extend(self._recommend_loop_fixes(diagnostic_analysis))

        # 4. Recommend removal of inefficient simp attributes
        recommendations.extend(self._recommend_remove_inefficient(diagnostic_analysis))

        # 5. Recommend lemma reordering optimizations
        recommendations.extend(self._recommend_reordering(diagnostic_analysis))

        plan = OptimizationPlan(recommendations=recommendations)

        logger.info(f"Generated {plan.total_recommendations} optimization recommendations:")
        logger.info(f"  High confidence: {len(plan.high_confidence)}")
        logger.info(f"  Medium confidence: {len(plan.medium_confidence)}")
        logger.info(f"  Low confidence: {len(plan.low_confidence)}")

        return plan

    def _scan_simp_rules(self) -> None:
        """Scan codebase for simp rule definitions."""
        self.simp_rules_cache.clear()

        # Pattern to match @[simp] and similar attributes
        simp_pattern = re.compile(
            r"@\[simp(?:\s+(\d+))?\]\s*(?:theorem|lemma|def)\s+(\w+)", re.MULTILINE
        )

        lean_files = list(self.project_path.glob("**/*.lean"))

        for file_path in lean_files:
            try:
                content = file_path.read_text(encoding="utf-8")

                for match in simp_pattern.finditer(content):
                    priority_str = match.group(1)
                    theorem_name = match.group(2)

                    # Calculate line number
                    line_num = content[: match.start()].count("\n") + 1
                    priority = int(priority_str) if priority_str else 1000

                    self.simp_rules_cache[theorem_name] = (file_path, line_num, priority)

            except Exception as e:
                logger.warning(f"Failed to scan {file_path}: {e}")

        logger.info(f"Found {len(self.simp_rules_cache)} simp rules in codebase")

    def _recommend_priority_increases(
        self, analysis: DiagnosticAnalysis
    ) -> list[OptimizationRecommendation]:
        """Recommend priority increases for frequently used theorems."""
        recommendations = []

        # Get most frequently used theorems
        most_used = analysis.get_most_used_theorems(limit=20)

        for theorem in most_used:
            if theorem.name not in self.simp_rules_cache:
                continue

            file_path, line_num, current_priority = self.simp_rules_cache[theorem.name]

            # Calculate evidence score based on usage frequency and success rate
            usage_score = min(100, theorem.used_count * 2)  # Cap at 100
            efficiency_score = theorem.success_rate * 100
            evidence_score = (usage_score + efficiency_score) / 2

            # Only recommend if theorem is used frequently and efficiently
            if theorem.used_count >= 10 and theorem.success_rate >= 0.8:
                # Calculate new priority: higher usage = lower number (higher priority)
                new_priority = max(50, 1000 - (theorem.used_count * 5))

                if new_priority < current_priority:
                    recommendations.append(
                        OptimizationRecommendation(
                            theorem_name=theorem.name,
                            file_path=file_path,
                            line_number=line_num,
                            optimization_type=OptimizationType.PRIORITY_INCREASE,
                            current_priority=current_priority,
                            recommended_priority=new_priority,
                            evidence_score=evidence_score,
                            reason=f"Used {theorem.used_count} times with {theorem.success_rate:.1%} success rate",
                            expected_impact="Faster simp by trying this theorem earlier",
                        )
                    )

        return recommendations

    def _recommend_priority_decreases(
        self, analysis: DiagnosticAnalysis
    ) -> list[OptimizationRecommendation]:
        """Recommend priority decreases for inefficient theorems."""
        recommendations = []

        # Get least efficient theorems
        inefficient = analysis.get_least_efficient_theorems(limit=10)

        for theorem in inefficient:
            if theorem.name not in self.simp_rules_cache:
                continue

            file_path, line_num, current_priority = self.simp_rules_cache[theorem.name]

            # Calculate evidence score based on inefficiency
            inefficiency_score = (1 - theorem.success_rate) * 100
            frequency_penalty = min(
                50, theorem.tried_count / 10
            )  # Penalty for being tried often but failing
            evidence_score = min(100, inefficiency_score + frequency_penalty)

            # Only recommend if theorem has poor success rate and is tried often
            if theorem.tried_count >= 20 and theorem.success_rate < 0.3:
                # Lower priority (higher number) for inefficient theorems
                new_priority = min(2000, current_priority + 500)

                recommendations.append(
                    OptimizationRecommendation(
                        theorem_name=theorem.name,
                        file_path=file_path,
                        line_number=line_num,
                        optimization_type=OptimizationType.PRIORITY_DECREASE,
                        current_priority=current_priority,
                        recommended_priority=new_priority,
                        evidence_score=evidence_score,
                        reason=f"Tried {theorem.tried_count} times but only {theorem.success_rate:.1%} success rate",
                        expected_impact="Avoid wasting time on ineffective theorem",
                    )
                )

        return recommendations

    def _recommend_loop_fixes(
        self, analysis: DiagnosticAnalysis
    ) -> list[OptimizationRecommendation]:
        """Recommend fixes for looping simp theorems."""
        recommendations = []

        for theorem_name in analysis.looping_theorems:
            if theorem_name not in self.simp_rules_cache:
                continue

            file_path, line_num, current_priority = self.simp_rules_cache[theorem_name]
            theorem = analysis.simp_theorems[theorem_name]

            # High evidence score for potential loops
            evidence_score = min(100, theorem.used_count / 5)

            recommendations.append(
                OptimizationRecommendation(
                    theorem_name=theorem_name,
                    file_path=file_path,
                    line_number=line_num,
                    optimization_type=OptimizationType.FIX_LOOP,
                    current_priority=current_priority,
                    recommended_priority=None,  # Manual fix required
                    evidence_score=evidence_score,
                    reason=f"Potential loop detected: used {theorem.used_count} times",
                    expected_impact="Prevent infinite simp loops and timeouts",
                )
            )

        return recommendations

    def _recommend_remove_inefficient(
        self, analysis: DiagnosticAnalysis
    ) -> list[OptimizationRecommendation]:
        """Recommend removing simp attribute from very inefficient theorems."""
        recommendations = []

        for theorem in analysis.simp_theorems.values():
            if theorem.name not in self.simp_rules_cache:
                continue

            # Consider removing simp attribute if theorem is almost never successful
            if theorem.tried_count >= 50 and theorem.success_rate < 0.05:
                file_path, line_num, current_priority = self.simp_rules_cache[theorem.name]

                evidence_score = (1 - theorem.success_rate) * 100

                recommendations.append(
                    OptimizationRecommendation(
                        theorem_name=theorem.name,
                        file_path=file_path,
                        line_number=line_num,
                        optimization_type=OptimizationType.REMOVE_INEFFICIENT,
                        current_priority=current_priority,
                        recommended_priority=None,  # Remove entirely
                        evidence_score=evidence_score,
                        reason=f"Extremely low success rate: {theorem.success_rate:.1%} over {theorem.tried_count} attempts",
                        expected_impact="Reduce simp overhead by removing ineffective theorem",
                    )
                )

        return recommendations

    def _recommend_reordering(
        self, analysis: DiagnosticAnalysis
    ) -> list[OptimizationRecommendation]:
        """Recommend reordering optimizations for manually specified simp sets."""
        recommendations = []

        # This would require more sophisticated analysis of simp call patterns
        # For now, we focus on the priority-based optimizations
        # Future enhancement: analyze `simp [list, of, lemmas]` patterns

        return recommendations

    def apply_recommendation(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply a single optimization recommendation."""
        try:
            content = recommendation.file_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            if (
                recommendation.optimization_type == OptimizationType.PRIORITY_INCREASE
                or recommendation.optimization_type == OptimizationType.PRIORITY_DECREASE
            ):
                return self._apply_priority_change(recommendation, lines)

            elif recommendation.optimization_type == OptimizationType.REMOVE_INEFFICIENT:
                return self._apply_remove_simp_attribute(recommendation, lines)

            elif recommendation.optimization_type == OptimizationType.FIX_LOOP:
                # Loop fixes require manual intervention
                logger.warning(
                    f"Loop fix for {recommendation.theorem_name} requires manual intervention"
                )
                return False

            return False

        except Exception as e:
            logger.error(f"Failed to apply recommendation for {recommendation.theorem_name}: {e}")
            return False

    def _apply_priority_change(
        self, recommendation: OptimizationRecommendation, lines: list[str]
    ) -> bool:
        """Apply priority change to a theorem."""
        line_index = recommendation.line_number - 1

        if line_index >= len(lines):
            logger.error(
                f"Invalid line number {recommendation.line_number} for {recommendation.file_path}"
            )
            return False

        # Find the @[simp] attribute (might be on current line or lines above)
        for i in range(max(0, line_index - 5), min(len(lines), line_index + 2)):
            line = lines[i]

            # Replace @[simp] or @[simp N] with @[simp new_priority]
            if "@[simp" in line:
                new_line = re.sub(
                    r"@\[simp(?:\s+\d+)?\]", f"@[simp {recommendation.recommended_priority}]", line
                )
                lines[i] = new_line

                # Write back to file
                new_content = "\n".join(lines)
                recommendation.file_path.write_text(new_content, encoding="utf-8")

                logger.info(
                    f"Applied priority change to {recommendation.theorem_name}: "
                    f"{recommendation.current_priority} → {recommendation.recommended_priority}"
                )
                return True

        logger.warning(f"Could not find @[simp] attribute for {recommendation.theorem_name}")
        return False

    def _apply_remove_simp_attribute(
        self, recommendation: OptimizationRecommendation, lines: list[str]
    ) -> bool:
        """Remove simp attribute from a theorem."""
        line_index = recommendation.line_number - 1

        # Find and remove the @[simp] attribute
        for i in range(max(0, line_index - 5), min(len(lines), line_index + 2)):
            line = lines[i]

            if "@[simp" in line:
                # Remove the @[simp] attribute
                new_line = re.sub(r"@\[simp(?:\s+\d+)?\]\s*", "", line)
                lines[i] = new_line

                # Write back to file
                new_content = "\n".join(lines)
                recommendation.file_path.write_text(new_content, encoding="utf-8")

                logger.info(f"Removed simp attribute from {recommendation.theorem_name}")
                return True

        logger.warning(
            f"Could not find @[simp] attribute to remove for {recommendation.theorem_name}"
        )
        return False

    def apply_plan(
        self, plan: OptimizationPlan, confidence_threshold: float = 50.0
    ) -> tuple[int, int]:
        """Apply optimization plan with confidence threshold."""
        applied = 0
        failed = 0

        recommendations_to_apply = [
            rec for rec in plan.recommendations if rec.evidence_score >= confidence_threshold
        ]

        logger.info(
            f"Applying {len(recommendations_to_apply)} recommendations with confidence >= {confidence_threshold}"
        )

        for recommendation in recommendations_to_apply:
            if self.apply_recommendation(recommendation):
                applied += 1
            else:
                failed += 1

        logger.info(f"Applied {applied} recommendations, {failed} failed")
        return applied, failed


# Example usage
if __name__ == "__main__":
    # This would be used with real diagnostic data
    from .diagnostic_parser import DiagnosticAnalysis, SimpTheoremUsage

    # Create sample analysis
    analysis = DiagnosticAnalysis()
    analysis.simp_theorems = {
        "frequently_used_theorem": SimpTheoremUsage("frequently_used_theorem", 150, 160, 150),
        "inefficient_theorem": SimpTheoremUsage("inefficient_theorem", 5, 200, 5),
        "looping_theorem": SimpTheoremUsage("looping_theorem", 500, 500, 500),
    }

    # This would use a real project path
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        engine = OptimizationEngine(Path(temp_dir))
        plan = engine.analyze_and_recommend(analysis)

        print(f"Generated {plan.total_recommendations} recommendations")
        for rec in plan.high_confidence:
            print(
                f"  {rec.theorem_name}: {rec.optimization_type.value} (confidence: {rec.evidence_score:.1f})"
            )
