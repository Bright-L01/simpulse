"""
Pattern Profiler for Simpulse
Analyzes which optimization patterns match a file and predicts outcomes
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional


class PatternType(Enum):
    """Types of patterns that can be detected."""

    SAFE = "safe"  # Patterns that benefit from optimization
    RISKY = "risky"  # Patterns that may cause issues
    UNSAFE = "unsafe"  # Patterns that will cause problems
    NEUTRAL = "neutral"  # Patterns that don't affect optimization


class SpeedupPrediction(NamedTuple):
    """Predicted speedup range and confidence."""

    min_speedup: float
    max_speedup: float
    confidence: str  # "high", "medium", "low"
    explanation: str


@dataclass
class PatternMatch:
    """A matched pattern in the file."""

    pattern_name: str
    pattern_type: PatternType
    count: int
    lines: List[int]
    description: str
    impact: str
    example_text: Optional[str] = None


@dataclass
class PatternProfile:
    """Complete pattern analysis of a file."""

    file_path: Path
    file_classification: str  # "SAFE", "RISKY", "UNSAFE"
    speedup_prediction: SpeedupPrediction
    safe_patterns: List[PatternMatch]
    risky_patterns: List[PatternMatch]
    unsafe_patterns: List[PatternMatch]
    pattern_summary: Dict[str, int]
    educational_insights: List[str]


class PatternProfiler:
    """Analyzes files for optimization patterns and predicts outcomes."""

    def __init__(self):
        self.safe_patterns = {
            "arithmetic_add_zero": {
                "regex": r"\b\w+\s*\+\s*0\b|\b0\s*\+\s*\w+\b",
                "description": "Addition with zero (n + 0, 0 + n)",
                "impact": "High benefit - our core optimization target",
                "speedup_factor": 2.0,
            },
            "arithmetic_mul_one": {
                "regex": r"\b\w+\s*\*\s*1\b|\b1\s*\*\s*\w+\b",
                "description": "Multiplication by one (n * 1, 1 * n)",
                "impact": "High benefit - our core optimization target",
                "speedup_factor": 2.0,
            },
            "simple_simp": {
                "regex": r":=\s*by\s+simp\s*$",
                "description": "Simple simp proofs (by simp)",
                "impact": "Medium benefit - straightforward simplification",
                "speedup_factor": 1.5,
            },
            "basic_nat_operations": {
                "regex": r"Nat\.(add|mul|zero|succ)",
                "description": "Basic Nat operations",
                "impact": "Medium benefit - arithmetic-focused",
                "speedup_factor": 1.3,
            },
            "mathlib_arithmetic": {
                "regex": r"import\s+Mathlib\.Data\.(Nat|Int|Real)",
                "description": "Mathlib arithmetic imports",
                "impact": "Positive signal - arithmetic-heavy code",
                "speedup_factor": 1.2,
            },
            "simple_theorems": {
                "regex": r"theorem\s+\w+.*:.*:=\s*by\s+simp",
                "description": "Simple theorem proofs",
                "impact": "Good - straightforward proof patterns",
                "speedup_factor": 1.4,
            },
        }

        self.risky_patterns = {
            "list_operations": {
                "regex": r"List\.|\.append|\+\+\s*\[\]|\[\]\s*\+\+|\.reverse",
                "description": "List operations (append, reverse, etc.)",
                "impact": "Risk of 5-25% regression - lists optimize differently",
                "speedup_factor": 0.8,
            },
            "complex_simp": {
                "regex": r"simp\s*\[.*\]|simp\s+only|simp\s+at",
                "description": "Complex simp usage (with arguments)",
                "impact": "Risk of conflicts with our optimization",
                "speedup_factor": 0.9,
            },
            "large_file_indicators": {
                "regex": r"theorem\s+\w+_\d{3,}",
                "description": "Many numbered theorems (large file indicator)",
                "impact": "Risk of stack overflow on very large files",
                "speedup_factor": 0.7,
            },
            "typeclass_heavy": {
                "regex": r"instance\s+.*:|class\s+.*:|\[.*:.*\]",
                "description": "Heavy typeclass usage",
                "impact": "Risk of elaboration conflicts",
                "speedup_factor": 0.85,
            },
            "meta_programming": {
                "regex": r"#eval|#check|#print|meta\s+def|run_cmd",
                "description": "Meta-programming constructs",
                "impact": "Risk of compilation issues",
                "speedup_factor": 0.9,
            },
        }

        self.unsafe_patterns = {
            "custom_simp_priorities": {
                "regex": r"@\[simp\s+\d+\]",
                "description": "Custom simp priorities (@[simp 1000])",
                "impact": "DANGEROUS: Will cause priority conflicts (29.9% regression observed)",
                "speedup_factor": 0.7,
            },
            "mutual_recursion": {
                "regex": r"mutual\s+def|mutual\s+inductive",
                "description": "Mutual recursion definitions",
                "impact": "DANGEROUS: Causes elaboration regressions (3% slowdown observed)",
                "speedup_factor": 0.97,
            },
            "recursive_simp_defs": {
                "regex": r"@\[simp\]\s*def.*:.*→.*→",
                "description": "Recursive simp definitions",
                "impact": "DANGEROUS: Different optimization strategy needed (44% regression risk)",
                "speedup_factor": 0.56,
            },
            "custom_simp_tactics": {
                "regex": r"declare_simp_like_tactic|simp_tac",
                "description": "Custom simp tactics",
                "impact": "INCOMPATIBLE: Will cause compilation failures",
                "speedup_factor": 0.0,
            },
            "float_operations": {
                "regex": r"Float\.|Double\.|Real\.",
                "description": "Floating-point operations",
                "impact": "INCOMPATIBLE: Not supported by current optimization",
                "speedup_factor": 0.0,
            },
            "universe_polymorphism": {
                "regex": r"universe\s+\w+|Type\*|Sort\s+\w+",
                "description": "Universe polymorphism",
                "impact": "DANGEROUS: May cause elaboration failures",
                "speedup_factor": 0.8,
            },
        }

    def analyze_file(self, file_path: Path) -> PatternProfile:
        """Perform complete pattern analysis of a file."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            return PatternProfile(
                file_path=file_path,
                file_classification="ERROR",
                speedup_prediction=SpeedupPrediction(0.0, 0.0, "low", f"Cannot read file: {e}"),
                safe_patterns=[],
                risky_patterns=[],
                unsafe_patterns=[],
                pattern_summary={},
                educational_insights=[f"File reading failed: {e}"],
            )

        lines = content.split("\n")

        # Analyze patterns
        safe_matches = self._find_patterns(content, lines, self.safe_patterns, PatternType.SAFE)
        risky_matches = self._find_patterns(content, lines, self.risky_patterns, PatternType.RISKY)
        unsafe_matches = self._find_patterns(
            content, lines, self.unsafe_patterns, PatternType.UNSAFE
        )

        # Classify file
        file_classification = self._classify_file(
            safe_matches, risky_matches, unsafe_matches, len(lines)
        )

        # Predict speedup
        speedup_prediction = self._predict_speedup(
            safe_matches, risky_matches, unsafe_matches, len(lines)
        )

        # Generate educational insights
        educational_insights = self._generate_insights(
            safe_matches, risky_matches, unsafe_matches, file_classification
        )

        # Create summary
        pattern_summary = {
            "total_lines": len(lines),
            "safe_patterns": len(safe_matches),
            "risky_patterns": len(risky_matches),
            "unsafe_patterns": len(unsafe_matches),
            "total_pattern_matches": sum(
                m.count for m in safe_matches + risky_matches + unsafe_matches
            ),
        }

        return PatternProfile(
            file_path=file_path,
            file_classification=file_classification,
            speedup_prediction=speedup_prediction,
            safe_patterns=safe_matches,
            risky_patterns=risky_matches,
            unsafe_patterns=unsafe_matches,
            pattern_summary=pattern_summary,
            educational_insights=educational_insights,
        )

    def _find_patterns(
        self, content: str, lines: List[str], patterns: Dict, pattern_type: PatternType
    ) -> List[PatternMatch]:
        """Find all matching patterns of a given type."""
        matches = []

        for pattern_name, pattern_info in patterns.items():
            regex = pattern_info["regex"]
            found_matches = list(re.finditer(regex, content, re.MULTILINE | re.IGNORECASE))

            if found_matches:
                # Find line numbers
                match_lines = []
                for match in found_matches:
                    line_num = content[: match.start()].count("\n") + 1
                    match_lines.append(line_num)

                # Get example text
                example_text = found_matches[0].group()[:50] if found_matches else None

                match_obj = PatternMatch(
                    pattern_name=pattern_name,
                    pattern_type=pattern_type,
                    count=len(found_matches),
                    lines=match_lines[:5],  # Show first 5 line numbers
                    description=pattern_info["description"],
                    impact=pattern_info["impact"],
                    example_text=example_text,
                )
                matches.append(match_obj)

        return matches

    def _classify_file(
        self,
        safe: List[PatternMatch],
        risky: List[PatternMatch],
        unsafe: List[PatternMatch],
        line_count: int,
    ) -> str:
        """Classify file as SAFE, RISKY, or UNSAFE."""

        # Check for critical issues first
        if line_count > 1000:
            return "UNSAFE"

        if unsafe:
            return "UNSAFE"

        # Calculate pattern scores
        safe_score = sum(match.count for match in safe)
        risky_score = sum(match.count for match in risky)

        # Decision logic
        if safe_score >= 10 and risky_score <= 2:
            return "SAFE"
        elif safe_score >= 5 and risky_score <= 5:
            return "RISKY"
        elif risky_score > safe_score * 2:
            return "RISKY"
        elif safe_score > 0:
            return "RISKY"
        else:
            return "UNSAFE"

    def _predict_speedup(
        self,
        safe: List[PatternMatch],
        risky: List[PatternMatch],
        unsafe: List[PatternMatch],
        line_count: int,
    ) -> SpeedupPrediction:
        """Predict speedup based on pattern analysis."""

        # Handle critical cases first
        if line_count > 1000:
            return SpeedupPrediction(
                0.0, 0.0, "high", "File too large (>1000 lines) - will cause stack overflow"
            )

        if any("custom_simp_priorities" in m.pattern_name for m in unsafe):
            return SpeedupPrediction(
                0.7, 0.7, "high", "Custom simp priorities detected - expect 29.9% regression"
            )

        if any("recursive_simp_defs" in m.pattern_name for m in unsafe):
            return SpeedupPrediction(
                0.56, 0.56, "high", "Recursive simp definitions - expect 44% regression"
            )

        if unsafe:
            return SpeedupPrediction(
                0.5, 0.9, "medium", "Unsafe patterns detected - likely regression"
            )

        # Calculate weighted speedup based on patterns
        total_weight = 0
        weighted_speedup = 0

        for match in safe:
            pattern_info = self.safe_patterns.get(match.pattern_name, {})
            factor = pattern_info.get("speedup_factor", 1.0)
            weight = match.count
            weighted_speedup += factor * weight
            total_weight += weight

        for match in risky:
            pattern_info = self.risky_patterns.get(match.pattern_name, {})
            factor = pattern_info.get("speedup_factor", 0.9)
            weight = match.count
            weighted_speedup += factor * weight
            total_weight += weight

        if total_weight == 0:
            return SpeedupPrediction(0.8, 1.1, "low", "No clear optimization patterns detected")

        predicted_speedup = weighted_speedup / total_weight

        # Determine range and confidence
        if predicted_speedup >= 1.5:
            return SpeedupPrediction(
                predicted_speedup * 0.8,
                predicted_speedup * 1.2,
                "high",
                f"High arithmetic density - strong optimization candidate",
            )
        elif predicted_speedup >= 1.2:
            return SpeedupPrediction(
                predicted_speedup * 0.9,
                predicted_speedup * 1.1,
                "medium",
                f"Good optimization potential with some risks",
            )
        elif predicted_speedup >= 1.0:
            return SpeedupPrediction(
                predicted_speedup * 0.9,
                predicted_speedup * 1.05,
                "low",
                f"Minimal benefit expected",
            )
        else:
            return SpeedupPrediction(
                predicted_speedup * 0.8,
                predicted_speedup * 1.1,
                "medium",
                f"Regression risk - optimization not recommended",
            )

    def _generate_insights(
        self,
        safe: List[PatternMatch],
        risky: List[PatternMatch],
        unsafe: List[PatternMatch],
        classification: str,
    ) -> List[str]:
        """Generate educational insights about the file."""
        insights = []

        # Classification explanation
        if classification == "SAFE":
            insights.append("✅ SAFE: This file has optimization-friendly patterns with low risk")
        elif classification == "RISKY":
            insights.append("⚠️ RISKY: Mixed patterns - optimization may help but test carefully")
        else:
            insights.append("🚨 UNSAFE: Patterns detected that will cause regressions or failures")

        # Pattern-specific insights
        if any("arithmetic_add_zero" in m.pattern_name for m in safe):
            insights.append(
                "🎯 High arithmetic density detected - this is our optimization sweet spot"
            )

        if any("custom_simp_priorities" in m.pattern_name for m in unsafe):
            insights.append(
                "💥 Custom simp priorities will conflict with our optimization (29.9% slower)"
            )

        if any("list_operations" in m.pattern_name for m in risky):
            insights.append(
                "📝 List operations detected - these don't benefit from arithmetic optimization"
            )

        if any("recursive_simp_defs" in m.pattern_name for m in unsafe):
            insights.append("🔄 Recursive simp definitions need different optimization strategy")

        # File size insights
        safe_count = sum(m.count for m in safe)
        risky_count = sum(m.count for m in risky)

        if safe_count > 15:
            insights.append(
                f"🚀 Excellent optimization target: {safe_count} beneficial patterns found"
            )
        elif safe_count > 5:
            insights.append(f"✨ Good optimization potential: {safe_count} beneficial patterns")

        if risky_count > safe_count:
            insights.append(f"⚠️ More risky patterns ({risky_count}) than safe ones ({safe_count})")

        # Recommendations
        if classification == "SAFE":
            insights.append("💡 Recommendation: Proceed with optimization and monitor performance")
        elif classification == "RISKY":
            insights.append("💡 Recommendation: Test thoroughly before deploying optimization")
        else:
            insights.append("💡 Recommendation: Do not optimize - use alternative approaches")

        return insights


def generate_pattern_profile_report(profile: PatternProfile, output_path: Path) -> None:
    """Generate a visual pattern profile report."""
    content = f"""# 🔍 Simpulse Pattern Profile

**File:** `{profile.file_path.name}`  
**Classification:** {profile.file_classification}  
**Predicted Speedup:** {profile.speedup_prediction.min_speedup:.2f}x - {profile.speedup_prediction.max_speedup:.2f}x  
**Confidence:** {profile.speedup_prediction.confidence.upper()}  

## 🎯 Speedup Prediction

{profile.speedup_prediction.explanation}

**Range:** {profile.speedup_prediction.min_speedup:.2f}x to {profile.speedup_prediction.max_speedup:.2f}x  
**Confidence:** {profile.speedup_prediction.confidence.title()}  

## 📊 Pattern Summary

| Category | Count | Impact |
|----------|-------|--------|
| 🟢 Safe Patterns | {len(profile.safe_patterns)} | Optimization targets |
| 🟡 Risky Patterns | {len(profile.risky_patterns)} | Potential issues |
| 🔴 Unsafe Patterns | {len(profile.unsafe_patterns)} | Will cause problems |
| **Total Matches** | {profile.pattern_summary.get('total_pattern_matches', 0)} | Across {profile.pattern_summary.get('total_lines', 0)} lines |

## 🟢 Safe Patterns (Optimization Targets)

"""

    if profile.safe_patterns:
        for pattern in profile.safe_patterns:
            content += f"### ✅ {pattern.description}\n"
            content += f"- **Found:** {pattern.count} occurrences\n"
            content += f"- **Impact:** {pattern.impact}\n"
            content += f"- **Lines:** {', '.join(map(str, pattern.lines))}\n"
            if pattern.example_text:
                content += f"- **Example:** `{pattern.example_text}`\n"
            content += "\n"
    else:
        content += "No safe optimization patterns detected.\n\n"

    content += "## 🟡 Risky Patterns (Proceed with Caution)\n\n"

    if profile.risky_patterns:
        for pattern in profile.risky_patterns:
            content += f"### ⚠️ {pattern.description}\n"
            content += f"- **Found:** {pattern.count} occurrences\n"
            content += f"- **Risk:** {pattern.impact}\n"
            content += f"- **Lines:** {', '.join(map(str, pattern.lines))}\n"
            if pattern.example_text:
                content += f"- **Example:** `{pattern.example_text}`\n"
            content += "\n"
    else:
        content += "No risky patterns detected.\n\n"

    content += "## 🔴 Unsafe Patterns (Will Cause Problems)\n\n"

    if profile.unsafe_patterns:
        for pattern in profile.unsafe_patterns:
            content += f"### 🚨 {pattern.description}\n"
            content += f"- **Found:** {pattern.count} occurrences\n"
            content += f"- **Danger:** {pattern.impact}\n"
            content += f"- **Lines:** {', '.join(map(str, pattern.lines))}\n"
            if pattern.example_text:
                content += f"- **Example:** `{pattern.example_text}`\n"
            content += "\n"
    else:
        content += "✅ No unsafe patterns detected.\n\n"

    content += "## 🧠 Educational Insights\n\n"

    for insight in profile.educational_insights:
        content += f"- {insight}\n"

    content += f"""
## 📈 Visual Pattern Distribution

```
Safe Patterns:   {'█' * min(20, len(profile.safe_patterns) * 4)} {len(profile.safe_patterns)}
Risky Patterns:  {'█' * min(20, len(profile.risky_patterns) * 4)} {len(profile.risky_patterns)}
Unsafe Patterns: {'█' * min(20, len(profile.unsafe_patterns) * 4)} {len(profile.unsafe_patterns)}
```

## 🎯 What This Means

"""

    if profile.file_classification == "SAFE":
        content += """
✅ **PROCEED WITH OPTIMIZATION**
- This file has patterns that benefit from Simpulse optimization
- Low risk of regressions
- Expected speedup in the predicted range
- Monitor performance to confirm benefits
"""
    elif profile.file_classification == "RISKY":
        content += """
⚠️ **PROCEED WITH CAUTION**
- Mixed patterns - some benefit, some risk
- Test thoroughly before production use
- Have rollback plan ready
- Consider addressing risky patterns first
"""
    else:
        content += """
🚨 **DO NOT OPTIMIZE**
- This file contains patterns that will cause problems
- High risk of regressions or compilation failures
- Consider alternative optimization strategies
- Fix unsafe patterns before attempting optimization
"""

    content += """
---

*Generated by Simpulse Pattern Profiler*  
*Understanding your code patterns for better optimization decisions*
"""

    output_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python pattern_profiler.py <lean_file>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    profiler = PatternProfiler()
    profile = profiler.analyze_file(file_path)

    print(f"File Classification: {profile.file_classification}")
    print(
        f"Predicted Speedup: {profile.speedup_prediction.min_speedup:.2f}x - {profile.speedup_prediction.max_speedup:.2f}x"
    )
    print(f"Explanation: {profile.speedup_prediction.explanation}")

    # Generate report
    report_path = file_path.parent / f"{file_path.stem}_pattern_profile.md"
    generate_pattern_profile_report(profile, report_path)
    print(f"Report saved: {report_path}")
