#!/usr/bin/env python3
"""
Success Predictor - Predicts optimization success based on pattern analysis

Key insight: Mixed pattern files have only 15% success rate (essentially random).
This module helps Simpulse skip files that are unlikely to benefit.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from simpulse.analysis.pattern_interference_analyzer import PatternInterferenceAnalyzer
from simpulse.analysis.sophisticated_pattern_analyzer import SophisticatedPatternAnalyzer

logger = logging.getLogger(__name__)


class OptimizationSuccessPredictor:
    """
    Predicts whether Simpulse optimization will succeed on a given file.

    Based on extensive analysis showing:
    - Mixed pattern files: 15% success (random)
    - Pure pattern files: 25-30% success
    - High interference files: <10% success
    """

    # Thresholds based on empirical analysis
    INTERFERENCE_THRESHOLD = 0.3
    CRITICAL_PAIRS_THRESHOLD = 10
    PATTERN_DIVERSITY_THRESHOLD = 0.8
    MINIMUM_SUCCESS_RATE = 0.20  # Skip if predicted success < 20%

    def __init__(self):
        self.interference_analyzer = PatternInterferenceAnalyzer()
        self.pattern_analyzer = SophisticatedPatternAnalyzer()

    def should_optimize(self, file_path: Path) -> Tuple[bool, Dict[str, any]]:
        """
        Determine if a file should be optimized.

        Returns:
            (should_optimize, reasoning_dict)
        """
        try:
            # Quick size check first
            file_size = file_path.stat().st_size
            if file_size > 100_000:  # 100KB
                return False, {
                    "reason": "File too large",
                    "details": f"File size {file_size} bytes exceeds 100KB limit",
                }

            # Analyze patterns and interference
            interference_result = self.interference_analyzer.analyze_file(file_path)
            pattern_result = self.pattern_analyzer.analyze_file(file_path)

            # Extract metrics
            metrics = interference_result["metrics"]
            interference_score = metrics["interference_score"]
            critical_pairs = metrics["critical_pairs"]
            pattern_diversity = metrics["pattern_diversity_index"]
            loop_risks = metrics["loop_risks"]

            # Predict success rate
            success_rate, reasoning = self._predict_success_rate(
                interference_score,
                critical_pairs,
                pattern_diversity,
                loop_risks,
                pattern_result["dominant_patterns"],
            )

            # Decision
            should_optimize = success_rate >= self.MINIMUM_SUCCESS_RATE

            return should_optimize, {
                "predicted_success_rate": success_rate,
                "reasoning": reasoning,
                "metrics": {
                    "interference_score": interference_score,
                    "critical_pairs": critical_pairs,
                    "pattern_diversity": pattern_diversity,
                    "loop_risks": loop_risks,
                },
                "recommendation": "optimize" if should_optimize else "skip",
            }

        except Exception as e:
            logger.warning(f"Error analyzing {file_path}: {e}")
            return False, {"reason": "Analysis failed", "error": str(e)}

    def _predict_success_rate(
        self,
        interference_score: float,
        critical_pairs: int,
        pattern_diversity: float,
        loop_risks: int,
        dominant_patterns: Dict[str, float],
    ) -> Tuple[float, str]:
        """
        Predict success rate based on metrics.

        Returns:
            (success_rate, reasoning)
        """
        # Hard failures
        if loop_risks > 0:
            return 0.0, "Loop risks detected - optimization will likely fail"

        if interference_score > 0.6:
            return 0.05, "Very high interference - optimization extremely unlikely to succeed"

        # Mixed patterns detection
        if pattern_diversity > self.PATTERN_DIVERSITY_THRESHOLD:
            # High diversity = mixed patterns = 15% success (random)
            return 0.15, "Mixed pattern file - success is essentially random (15%)"

        # High interference
        if interference_score > self.INTERFERENCE_THRESHOLD:
            return (
                0.10,
                f"High interference score ({interference_score:.2f}) - low success probability",
            )

        # Many critical pairs
        if critical_pairs > self.CRITICAL_PAIRS_THRESHOLD:
            return 0.12, f"Many critical pairs ({critical_pairs}) - high conflict potential"

        # Check for dominant pattern types
        identity_percentage = dominant_patterns.get("identity_patterns", 0)
        operator_percentage = dominant_patterns.get("operator_patterns", 0)

        # Pure arithmetic patterns
        if identity_percentage > 5 and operator_percentage > 10:
            return 0.30, "Pure arithmetic patterns - moderate success probability"

        # Default for reasonably uniform files
        return 0.25, "Relatively uniform patterns - reasonable success probability"

    def analyze_directory(self, directory: Path, sample_size: Optional[int] = None) -> Dict:
        """
        Analyze all Lean files in a directory and summarize optimization potential.
        """
        lean_files = list(directory.glob("**/*.lean"))
        if sample_size:
            lean_files = lean_files[:sample_size]

        results = []
        for file_path in lean_files:
            should_opt, reasoning = self.should_optimize(file_path)
            results.append(
                {
                    "file": str(file_path),
                    "should_optimize": should_opt,
                    "predicted_success_rate": reasoning.get("predicted_success_rate", 0),
                    "reason": reasoning.get("reasoning", ""),
                }
            )

        # Summary statistics
        total = len(results)
        recommended = sum(1 for r in results if r["should_optimize"])
        avg_success_rate = (
            sum(r["predicted_success_rate"] for r in results) / total if total > 0 else 0
        )

        return {
            "total_files": total,
            "recommended_for_optimization": recommended,
            "skip_recommended": total - recommended,
            "optimization_percentage": recommended / total * 100 if total > 0 else 0,
            "average_predicted_success_rate": avg_success_rate,
            "file_results": results,
        }


def main():
    """Example usage"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python success_predictor.py <lean_file_or_directory>")
        sys.exit(1)

    path = Path(sys.argv[1])
    predictor = OptimizationSuccessPredictor()

    if path.is_file():
        should_optimize, reasoning = predictor.should_optimize(path)
        print(f"File: {path}")
        print(f"Should optimize: {should_optimize}")
        print(f"Predicted success rate: {reasoning.get('predicted_success_rate', 0):.1%}")
        print(f"Reasoning: {reasoning.get('reasoning', 'Unknown')}")
        if "metrics" in reasoning:
            print(f"Metrics: {reasoning['metrics']}")

    elif path.is_dir():
        print(f"Analyzing directory: {path}")
        summary = predictor.analyze_directory(path, sample_size=10)
        print(f"\nSummary:")
        print(f"Total files: {summary['total_files']}")
        print(
            f"Recommended for optimization: {summary['recommended_for_optimization']} ({summary['optimization_percentage']:.1f}%)"
        )
        print(f"Average predicted success rate: {summary['average_predicted_success_rate']:.1%}")

        print(f"\nSample results:")
        for result in summary["file_results"][:5]:
            status = "✓" if result["should_optimize"] else "✗"
            print(
                f"{status} {result['file']}: {result['predicted_success_rate']:.1%} - {result['reason']}"
            )


if __name__ == "__main__":
    main()
