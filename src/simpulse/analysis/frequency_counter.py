"""
Real frequency counter for Lean 4 simp traces.
Parses actual compilation logs to count simp lemma usage.
No fake data, no simulations - only real trace analysis.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LemmaFrequency:
    """Frequency data for a single lemma."""

    lemma_name: str
    success_count: int = 0
    try_count: int = 0

    @property
    def total_count(self) -> int:
        return self.success_count + self.try_count

    @property
    def success_rate(self) -> float:
        if self.try_count == 0:
            return 0.0
        return self.success_count / self.try_count


class FrequencyCounter:
    """Parse real Lean 4 traces to count simp lemma usage."""

    # Real trace patterns from actual Lean 4 compilation
    TRACE_PATTERNS = {
        # Success pattern: lemma was successfully applied
        "success": re.compile(
            r"(?:\[trace\.(?:Meta\.)?[Tt]actic\.simp(?:\.rewrite)?\])\s+([a-zA-Z0-9_.]+)\s*:\s*(.+?)\s*==>(?:simp)?\s*(.+)"
        ),
        # Try pattern: lemma was attempted
        "try_lemma": re.compile(
            r"(?:\[trace\.(?:Meta\.)?[Tt]actic\.simp(?:\.rewrite)?\])\s+(?:trying|apply)\s+(?:simp\s+lemma\s+)?([a-zA-Z0-9_.]+(?:\.[a-zA-Z0-9_]+)*)"
        ),
        # Alternative formats
        "apply_lemma": re.compile(r"(?:\[trace\.simp\])\s+apply\s+([a-zA-Z0-9_.]+)"),
        # File location pattern
        "file_location": re.compile(r"--\s*File:\s*([^:]+):(\d+)"),
    }

    def __init__(self):
        self.frequencies: Dict[str, LemmaFrequency] = {}
        self.current_file: Optional[str] = None
        self.current_line: Optional[int] = None

    def count_frequencies(self, trace_file: Path) -> Dict[str, Dict[str, Any]]:
        """
        Count frequencies from a real Lean trace file.

        Returns dict mapping lemma names to frequency data.
        """
        if not trace_file.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_file}")

        self.frequencies = {}

        with open(trace_file, encoding="utf-8") as f:
            for line in f:
                self._process_trace_line(line.strip())

        # Convert to simple dict format
        return {
            name: {
                "success_count": freq.success_count,
                "try_count": freq.try_count,
                "total_count": freq.total_count,
                "success_rate": freq.success_rate,
            }
            for name, freq in self.frequencies.items()
        }

    def _process_trace_line(self, line: str):
        """Process a single trace line."""
        # Check for file location
        file_match = self.TRACE_PATTERNS["file_location"].search(line)
        if file_match:
            self.current_file = file_match.group(1)
            self.current_line = int(file_match.group(2))
            return

        # Parse trace patterns
        matches = self._parse_trace_line(line)
        for match in matches:
            lemma_name = match["lemma"]

            if lemma_name not in self.frequencies:
                self.frequencies[lemma_name] = LemmaFrequency(lemma_name)

            if match["type"] == "success":
                self.frequencies[lemma_name].success_count += 1
            elif match["type"] in ("try", "apply"):
                self.frequencies[lemma_name].try_count += 1

    def _parse_trace_line(self, line: str) -> List[Dict[str, str]]:
        """Parse a trace line and extract lemma information."""
        matches = []

        # Check success pattern
        success_match = self.TRACE_PATTERNS["success"].search(line)
        if success_match:
            matches.append(
                {
                    "type": "success",
                    "lemma": success_match.group(1),
                    "from": success_match.group(2),
                    "to": success_match.group(3),
                }
            )

        # Check try pattern
        try_match = self.TRACE_PATTERNS["try_lemma"].search(line)
        if try_match:
            matches.append({"type": "try", "lemma": try_match.group(1)})

        # Check apply pattern
        apply_match = self.TRACE_PATTERNS["apply_lemma"].search(line)
        if apply_match:
            matches.append({"type": "apply", "lemma": apply_match.group(1)})

        return matches

    def get_optimization_candidates(
        self, frequencies: Dict[str, Dict[str, Any]], top_n: int = 20
    ) -> List[Tuple[str, int]]:
        """
        Get top lemmas for optimization based on frequency.

        Returns list of (lemma_name, frequency) tuples.
        """
        # Sort by total count
        sorted_lemmas = sorted(
            frequencies.items(), key=lambda x: x[1].get("total_count", 0), reverse=True
        )

        return [(name, data["total_count"]) for name, data in sorted_lemmas[:top_n]]

    def estimate_optimization_impact(
        self, frequencies: Dict[str, Dict[str, Any]], top_n: int = 20
    ) -> Dict[str, Any]:
        """Estimate the impact of optimizing top lemmas."""
        if not frequencies:
            return {"estimated_reduction_percent": 0, "optimized_lemmas": []}

        total_attempts = sum(f.get("try_count", 0) for f in frequencies.values())
        if total_attempts == 0:
            return {"estimated_reduction_percent": 0, "optimized_lemmas": []}

        top_lemmas = self.get_optimization_candidates(frequencies, top_n)
        optimized_attempts = sum(count for _, count in top_lemmas)

        # Estimate reduction based on moving top lemmas to front
        # Assume average position improves from middle to front
        estimated_reduction = (optimized_attempts / total_attempts) * 0.5 * 100

        return {
            "estimated_reduction_percent": estimated_reduction,
            "optimized_lemmas": [lemma for lemma, _ in top_lemmas],
            "total_attempts": total_attempts,
            "optimized_attempts": optimized_attempts,
        }

    def find_hot_spots(self, trace_file: Path, threshold: int = 10) -> List[Tuple[str, int, int]]:
        """
        Find hot spots where many simp attempts occur.

        Returns list of (file, line, count) tuples.
        """
        location_counts = defaultdict(int)
        current_location = None

        with open(trace_file, encoding="utf-8") as f:
            for line in f:
                # Check for file location
                file_match = self.TRACE_PATTERNS["file_location"].search(line)
                if file_match:
                    current_location = (file_match.group(1), int(file_match.group(2)))
                    continue

                # Count simp attempts at current location
                if current_location and any(
                    pattern.search(line)
                    for pattern in self.TRACE_PATTERNS.values()
                    if pattern != self.TRACE_PATTERNS["file_location"]
                ):
                    location_counts[current_location] += 1

        # Filter by threshold and sort
        hot_spots = [
            (file, line, count)
            for (file, line), count in location_counts.items()
            if count >= threshold
        ]

        return sorted(hot_spots, key=lambda x: x[2], reverse=True)

    def generate_optimization_report(
        self, trace_file: Path, output_file: Optional[Path] = None
    ) -> str:
        """Generate a comprehensive optimization report."""
        frequencies = self.count_frequencies(trace_file)
        top_candidates = self.get_optimization_candidates(frequencies, top_n=50)
        impact = self.estimate_optimization_impact(frequencies)
        hot_spots = self.find_hot_spots(trace_file)

        report = f"""
SIMP FREQUENCY ANALYSIS REPORT
==============================

Trace File: {trace_file}
Total Lemmas: {len(frequencies)}
Total Attempts: {impact.get('total_attempts', 0)}

Top 20 Most Used Lemmas:
------------------------
"""
        for lemma, count in top_candidates[:20]:
            freq_data = frequencies[lemma]
            report += (
                f"  {lemma:<40} {count:>6} attempts ({freq_data['success_rate']:.1%} success)\n"
            )

        report += f"""
Optimization Impact:
-------------------
Estimated reduction: {impact['estimated_reduction_percent']:.1f}%
Lemmas to optimize: {len(impact['optimized_lemmas'])}

Hot Spots (>10 attempts):
------------------------
"""
        for file, line, count in hot_spots[:10]:
            report += f"  {file}:{line} - {count} attempts\n"

        report += (
            """
Recommended Optimization:
------------------------
attribute [simp 1200] """
            + " ".join(impact["optimized_lemmas"][:5])
            + """
attribute [simp 1199] """
            + " ".join(impact["optimized_lemmas"][5:10])
            + """
attribute [simp 1198] """
            + " ".join(impact["optimized_lemmas"][10:15])
            + "\n"
        )

        if output_file:
            output_file.write_text(report)

        return report


def main():
    """CLI entry point for frequency counter."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: frequency_counter.py <trace_file> [output_file]")
        sys.exit(1)

    trace_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    counter = FrequencyCounter()

    try:
        report = counter.generate_optimization_report(trace_file, output_file)
        print(report)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
