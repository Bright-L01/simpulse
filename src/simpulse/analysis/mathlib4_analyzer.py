"""
Analyze simp lemma usage patterns in mathlib4.

This tool examines the entire mathlib4 codebase to understand:
1. How many simp lemmas are defined
2. Their priority distributions
3. Usage patterns and potential optimizations
"""

import json
import logging
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simpulse.evolution.models import SimpPriority
from simpulse.evolution.rule_extractor import RuleExtractor

logger = logging.getLogger(__name__)


@dataclass
class SimpLemmaStats:
    """Statistics about a simp lemma."""

    name: str
    priority: int | str
    file: str
    line: int
    declaration_type: str  # theorem, lemma, def, etc.
    has_direction: bool
    attributes: List[str]  # Other attributes besides simp


@dataclass
class Mathlib4Analysis:
    """Complete analysis of mathlib4 simp lemmas."""

    total_lemmas: int
    total_files: int
    lemmas_by_priority: Dict[str, int]
    lemmas_by_file: Dict[str, List[str]]
    priority_distribution: Dict[str, int]
    direction_usage: Dict[str, int]
    multi_attribute_lemmas: int
    declaration_types: Dict[str, int]
    top_files: List[Tuple[str, int]]
    unusual_priorities: List[Tuple[str, str, str]]  # (lemma, priority, file)


class Mathlib4Analyzer:
    """Analyze simp lemma patterns in mathlib4."""

    def __init__(self, mathlib_path: Optional[Path] = None):
        """Initialize analyzer.

        Args:
            mathlib_path: Path to mathlib4 repository. If None, tries to find it.
        """
        self.mathlib_path = mathlib_path or self._find_mathlib4()
        self.extractor = RuleExtractor()
        self.lemma_stats: List[SimpLemmaStats] = []

    def _find_mathlib4(self) -> Path:
        """Try to find mathlib4 installation."""
        # Common locations
        candidates = [
            Path.home() / ".elan/packages/mathlib4",
            Path.home() / "mathlib4",
            Path("/usr/local/lib/lean/mathlib4"),
            Path.cwd() / "mathlib4",
            Path.cwd().parent / "mathlib4",
        ]

        for path in candidates:
            if path.exists() and (path / "Mathlib").exists():
                logger.info(f"Found mathlib4 at: {path}")
                return path

        # Try to find via lake
        try:
            result = subprocess.run(
                ["lake", "env", "printenv", "LEAN_SRC_PATH"], capture_output=True, text=True
            )
            if result.returncode == 0:
                for path_str in result.stdout.strip().split(":"):
                    path = Path(path_str)
                    if "mathlib4" in str(path) and path.exists():
                        return path.parent if path.name == "build" else path
        except:
            pass

        raise FileNotFoundError("Could not find mathlib4. Please specify path explicitly.")

    def analyze_all(self) -> Mathlib4Analysis:
        """Analyze all simp lemmas in mathlib4."""
        logger.info(f"Analyzing mathlib4 at: {self.mathlib_path}")

        # Collect all .lean files
        lean_files = list((self.mathlib_path / "Mathlib").rglob("*.lean"))
        logger.info(f"Found {len(lean_files)} Lean files to analyze")

        # Process each file
        for i, file_path in enumerate(lean_files):
            if i % 100 == 0:
                logger.info(f"Processing file {i}/{len(lean_files)}: {file_path.name}")

            try:
                self._analyze_file(file_path)
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")

        # Generate analysis
        return self._generate_analysis()

    def _analyze_file(self, file_path: Path):
        """Analyze simp lemmas in a single file."""
        # Use our rule extractor
        module_rules = self.extractor.extract_rules_from_file(file_path)

        # Convert to stats
        for rule in module_rules.rules:
            # Determine priority value
            if isinstance(rule.priority, SimpPriority):
                priority_str = rule.priority.value
                if priority_str == "high":
                    priority_val = 1500  # Approximate
                elif priority_str == "low":
                    priority_val = 500
                else:  # default
                    priority_val = 1000
            else:
                priority_val = rule.priority
                priority_str = str(rule.priority)

            # Extract other attributes from metadata
            other_attrs = []
            if "raw_attribute" in rule.metadata:
                # Parse for other attributes besides simp
                raw = rule.metadata["raw_attribute"]
                if "norm_cast" in raw:
                    other_attrs.append("norm_cast")
                if "to_additive" in raw:
                    other_attrs.append("to_additive")
                if "nolint" in raw:
                    other_attrs.append("nolint")

            stat = SimpLemmaStats(
                name=rule.name,
                priority=priority_val,
                file=str(file_path.relative_to(self.mathlib_path)),
                line=rule.location.line,
                declaration_type=rule.metadata.get("declaration_type", "unknown"),
                has_direction=rule.direction.value != "forward",
                attributes=other_attrs,
            )

            self.lemma_stats.append(stat)

    def _generate_analysis(self) -> Mathlib4Analysis:
        """Generate comprehensive analysis from collected stats."""
        # Basic counts
        total_lemmas = len(self.lemma_stats)
        files_with_lemmas = {stat.file for stat in self.lemma_stats}
        total_files = len(files_with_lemmas)

        # Priority distribution
        priority_counter = Counter()
        for stat in self.lemma_stats:
            if isinstance(stat.priority, int):
                if stat.priority == 1500:
                    priority_counter["high"] += 1
                elif stat.priority == 500:
                    priority_counter["low"] += 1
                elif stat.priority == 1000:
                    priority_counter["default"] += 1
                else:
                    priority_counter[f"custom_{stat.priority}"] += 1
            else:
                priority_counter[str(stat.priority)] += 1

        # Direction usage
        direction_counter = Counter()
        for stat in self.lemma_stats:
            if stat.has_direction:
                direction_counter["backward"] += 1
            else:
                direction_counter["forward"] += 1

        # Multi-attribute lemmas
        multi_attr_count = sum(1 for stat in self.lemma_stats if len(stat.attributes) > 0)

        # Declaration types
        decl_types = Counter(stat.declaration_type for stat in self.lemma_stats)

        # Files with most simp lemmas
        file_counter = Counter(stat.file for stat in self.lemma_stats)
        top_files = file_counter.most_common(20)

        # Find unusual priorities
        unusual = []
        for stat in self.lemma_stats:
            if isinstance(stat.priority, int) and stat.priority not in [500, 1000, 1500]:
                unusual.append((stat.name, str(stat.priority), stat.file))
            elif isinstance(stat.priority, str) and stat.priority not in ["default", "high", "low"]:
                unusual.append((stat.name, stat.priority, stat.file))

        # Sort by priority value for interesting cases
        unusual.sort(key=lambda x: (int(x[1]) if x[1].isdigit() else 0), reverse=True)

        # Group lemmas by priority
        lemmas_by_priority = defaultdict(list)
        for stat in self.lemma_stats:
            key = str(stat.priority)
            lemmas_by_priority[key].append(stat.name)

        # Group by file
        lemmas_by_file = defaultdict(list)
        for stat in self.lemma_stats:
            lemmas_by_file[stat.file].append(stat.name)

        return Mathlib4Analysis(
            total_lemmas=total_lemmas,
            total_files=total_files,
            lemmas_by_priority={k: len(v) for k, v in lemmas_by_priority.items()},
            lemmas_by_file=dict(lemmas_by_file),
            priority_distribution=dict(priority_counter),
            direction_usage=dict(direction_counter),
            multi_attribute_lemmas=multi_attr_count,
            declaration_types=dict(decl_types),
            top_files=top_files,
            unusual_priorities=unusual[:50],  # Top 50 unusual
        )

    def find_potential_issues(self) -> Dict[str, List[str]]:
        """Identify potential priority optimization opportunities."""
        issues = defaultdict(list)

        # Group lemmas by similar names (potential priority inconsistencies)
        name_groups = defaultdict(list)
        for stat in self.lemma_stats:
            # Extract base name (before underscore variations)
            base = stat.name.split("_")[0] if "_" in stat.name else stat.name
            name_groups[base].append(stat)

        # Find groups with inconsistent priorities
        for base_name, stats in name_groups.items():
            if len(stats) > 1:
                priorities = {stat.priority for stat in stats}
                if len(priorities) > 1:
                    issue = f"{base_name}*: inconsistent priorities {sorted(priorities)}"
                    issues["inconsistent_priorities"].append(issue)

        # Find potential priority inversions
        # (e.g., "basic" lemmas with high priority, "complex" with default)
        for stat in self.lemma_stats:
            # Simple heuristic: shorter names = more basic
            if len(stat.name) < 10 and stat.priority == 1500:
                issues["possibly_too_high"].append(f"{stat.name} (high priority)")
            elif len(stat.name) > 30 and stat.priority == 1000:
                issues["possibly_too_low"].append(f"{stat.name} (default priority)")

        # Find files with all default priorities (might benefit from curation)
        for file, lemmas in self.lemmas_by_file.items():
            file_stats = [s for s in self.lemma_stats if s.file == file]
            if len(file_stats) > 10:  # Only consider files with many lemmas
                priorities = {s.priority for s in file_stats}
                if priorities == {1000}:  # All default
                    issues["needs_priority_curation"].append(
                        f"{file}: {len(file_stats)} lemmas all with default priority"
                    )

        return dict(issues)

    def generate_usage_heatmap(self) -> Dict[str, any]:
        """Generate data for visualizing simp lemma distribution."""
        # Group by top-level module
        module_counts = defaultdict(int)
        for stat in self.lemma_stats:
            parts = stat.file.split("/")
            if parts[0] == "Mathlib" and len(parts) > 1:
                module = parts[1]
            else:
                module = parts[0]
            module_counts[module] += 1

        # Priority distribution by module
        module_priorities = defaultdict(lambda: defaultdict(int))
        for stat in self.lemma_stats:
            parts = stat.file.split("/")
            module = parts[1] if parts[0] == "Mathlib" and len(parts) > 1 else parts[0]
            priority_key = (
                "high" if stat.priority == 1500 else "low" if stat.priority == 500 else "default"
            )
            module_priorities[module][priority_key] += 1

        return {
            "module_counts": dict(module_counts),
            "module_priorities": {k: dict(v) for k, v in module_priorities.items()},
            "total_by_module": sorted(module_counts.items(), key=lambda x: x[1], reverse=True),
        }


def main():
    """Run mathlib4 analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze simp lemmas in mathlib4")
    parser.add_argument("--mathlib-path", type=Path, help="Path to mathlib4")
    parser.add_argument("--output", type=Path, default=Path("mathlib4_analysis.json"))
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        analyzer = Mathlib4Analyzer(args.mathlib_path)
        analysis = analyzer.analyze_all()

        # Print summary
        print("\n" + "=" * 60)
        print("MATHLIB4 SIMP LEMMA ANALYSIS")
        print("=" * 60)
        print(f"\nTotal simp lemmas: {analysis.total_lemmas:,}")
        print(f"Files with simp lemmas: {analysis.total_files:,}")
        print(f"Average lemmas per file: {analysis.total_lemmas/analysis.total_files:.1f}")

        print("\nPriority Distribution:")
        for priority, count in sorted(analysis.priority_distribution.items()):
            percentage = count / analysis.total_lemmas * 100
            print(f"  {priority:20} {count:6,} ({percentage:5.1f}%)")

        print("\nDirection Usage:")
        for direction, count in analysis.direction_usage.items():
            percentage = count / analysis.total_lemmas * 100
            print(f"  {direction:20} {count:6,} ({percentage:5.1f}%)")

        print("\nTop Files by Simp Lemma Count:")
        for file, count in analysis.top_files[:10]:
            print(f"  {count:4} {file}")

        print("\nDeclaration Types:")
        for decl_type, count in sorted(analysis.declaration_types.items()):
            percentage = count / analysis.total_lemmas * 100
            print(f"  {decl_type:20} {count:6,} ({percentage:5.1f}%)")

        if analysis.unusual_priorities:
            print("\nUnusual Priorities Found:")
            for lemma, priority, file in analysis.unusual_priorities[:10]:
                print(f"  {lemma}: priority {priority} in {file}")

        # Find issues
        issues = analyzer.find_potential_issues()
        if issues:
            print("\nPotential Optimization Opportunities:")
            for issue_type, examples in issues.items():
                print(f"\n{issue_type}:")
                for example in examples[:5]:
                    print(f"  - {example}")
                if len(examples) > 5:
                    print(f"  ... and {len(examples)-5} more")

        # Save detailed results
        results = {
            "summary": {
                "total_lemmas": analysis.total_lemmas,
                "total_files": analysis.total_files,
                "avg_per_file": analysis.total_lemmas / analysis.total_files,
            },
            "priority_distribution": analysis.priority_distribution,
            "direction_usage": analysis.direction_usage,
            "declaration_types": analysis.declaration_types,
            "top_files": analysis.top_files,
            "unusual_priorities": analysis.unusual_priorities[:100],
            "multi_attribute_count": analysis.multi_attribute_lemmas,
            "heatmap": analyzer.generate_usage_heatmap(),
        }

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nDetailed results saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
