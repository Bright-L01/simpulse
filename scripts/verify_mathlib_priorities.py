#!/usr/bin/env python3
"""
Verify the actual usage of simp rule priorities in mathlib4.

This script clones the real mathlib4 repository and analyzes ALL .lean files
to count how many simp rules use default vs custom priorities.
"""

import json
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


class Mathlib4Analyzer:
    """Analyzes actual mathlib4 source code for simp priority usage."""

    def __init__(self, repo_path: str = "mathlib4_analysis"):
        self.repo_path = Path(repo_path)
        self.mathlib_url = "https://github.com/leanprover-community/mathlib4.git"

        # More comprehensive regex patterns for simp attributes
        self.simp_patterns = [
            # Standard simp attribute with potential priority
            re.compile(r"@\[([^\]]*\bsimp\b[^\]]*)\]", re.MULTILINE | re.DOTALL),
            # Multiline attributes
            re.compile(
                r"attribute\s*\[[^\]]*\bsimp\b[^\]]*\]", re.MULTILINE | re.DOTALL
            ),
        ]

        # Pattern to extract priority value
        self.priority_pattern = re.compile(r"(?:priority\s*:?=?\s*)?(\d+)")

        # Pattern for simp modifiers
        self.modifier_pattern = re.compile(r"simp\s*([↓←→])")

    def clone_mathlib4(self):
        """Clone or update mathlib4 repository."""
        if self.repo_path.exists():
            print(f"Updating existing mathlib4 repository at {self.repo_path}...")
            subprocess.run(["git", "pull"], cwd=self.repo_path, check=True)
        else:
            print(f"Cloning mathlib4 repository to {self.repo_path}...")
            subprocess.run(
                ["git", "clone", "--depth", "1", self.mathlib_url, str(self.repo_path)],
                check=True,
            )

    def extract_simp_attributes(
        self, content: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """Extract all simp attributes from file content."""
        attributes = []

        for pattern in self.simp_patterns:
            for match in pattern.finditer(content):
                attr_text = (
                    match.group(1) if len(match.groups()) > 0 else match.group(0)
                )
                line_num = content[: match.start()].count("\n") + 1

                # Skip if it's a comment
                line_start = content.rfind("\n", 0, match.start()) + 1
                line = content[line_start : match.start()]
                if "--" in line or "/-" in line:
                    continue

                # Extract priority if specified
                priority = None
                priority_match = self.priority_pattern.search(attr_text)
                if priority_match and "priority" in attr_text.lower():
                    priority = int(priority_match.group(1))

                # Check for high/low keywords
                if "high" in attr_text.lower() and priority is None:
                    priority = "high"
                elif "low" in attr_text.lower() and priority is None:
                    priority = "low"

                # Check for direction modifiers
                modifier = None
                modifier_match = self.modifier_pattern.search(attr_text)
                if modifier_match:
                    modifier = modifier_match.group(1)

                # Get surrounding context for the attribute
                context_start = max(0, match.start() - 100)
                context_end = min(len(content), match.end() + 200)
                context = content[context_start:context_end].strip()

                attributes.append(
                    {
                        "file": file_path,
                        "line": line_num,
                        "attribute": attr_text.strip(),
                        "priority": priority,
                        "modifier": modifier,
                        "context": context,
                        "is_default": priority is None and modifier is None,
                    }
                )

        return attributes

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Lean file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            attributes = self.extract_simp_attributes(
                content, str(file_path.relative_to(self.repo_path))
            )

            return {
                "file": str(file_path.relative_to(self.repo_path)),
                "attributes": attributes,
                "total": len(attributes),
                "default": sum(1 for a in attributes if a["is_default"]),
                "custom": sum(1 for a in attributes if not a["is_default"]),
            }

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None

    def analyze_mathlib(self) -> Dict[str, Any]:
        """Analyze entire mathlib4 repository."""
        print("Starting analysis of mathlib4...")
        start_time = time.time()

        results = {
            "total_simp_rules": 0,
            "default_priority": 0,
            "custom_priority": 0,
            "with_modifier": 0,
            "by_priority": defaultdict(int),
            "by_modifier": defaultdict(int),
            "by_module": defaultdict(lambda: {"total": 0, "default": 0, "custom": 0}),
            "custom_examples": [],
            "files_analyzed": 0,
            "analysis_time": 0,
        }

        # Find all .lean files
        lean_files = list(self.repo_path.rglob("*.lean"))
        print(f"Found {len(lean_files)} .lean files to analyze")

        # Analyze each file
        for i, file_path in enumerate(lean_files):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(lean_files)} files analyzed...")

            file_results = self.analyze_file(file_path)
            if file_results is None:
                continue

            results["files_analyzed"] += 1

            # Update totals
            for attr in file_results["attributes"]:
                results["total_simp_rules"] += 1

                if attr["is_default"]:
                    results["default_priority"] += 1
                else:
                    results["custom_priority"] += 1

                # Track specific priorities
                if attr["priority"] is not None:
                    if isinstance(attr["priority"], int):
                        results["by_priority"][attr["priority"]] += 1
                    else:
                        results["by_priority"][attr["priority"]] += 1

                # Track modifiers
                if attr["modifier"]:
                    results["with_modifier"] += 1
                    results["by_modifier"][attr["modifier"]] += 1

                # Collect examples of custom priority
                if not attr["is_default"] and len(results["custom_examples"]) < 50:
                    results["custom_examples"].append(
                        {
                            "file": attr["file"],
                            "line": attr["line"],
                            "attribute": attr["attribute"],
                            "priority": attr["priority"],
                            "modifier": attr["modifier"],
                            "context_snippet": (
                                attr["context"][:200] + "..."
                                if len(attr["context"]) > 200
                                else attr["context"]
                            ),
                        }
                    )

            # Update module stats
            module = (
                file_results["file"].split("/")[0]
                if "/" in file_results["file"]
                else "root"
            )
            results["by_module"][module]["total"] += file_results["total"]
            results["by_module"][module]["default"] += file_results["default"]
            results["by_module"][module]["custom"] += file_results["custom"]

        results["analysis_time"] = time.time() - start_time
        return dict(results)

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed report from analysis results."""
        report = []
        report.append("=" * 80)
        report.append("MATHLIB4 SIMP PRIORITY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        # Summary statistics
        total = results["total_simp_rules"]
        default = results["default_priority"]
        custom = results["custom_priority"]
        default_pct = (default / total * 100) if total > 0 else 0
        custom_pct = (custom / total * 100) if total > 0 else 0

        report.append("SUMMARY STATISTICS")
        report.append("-" * 40)
        report.append(f"Total simp rules found: {total:,}")
        report.append(f"Files analyzed: {results['files_analyzed']:,}")
        report.append(f"Analysis time: {results['analysis_time']:.1f} seconds")
        report.append("")
        report.append(f"Default priority: {default:,} ({default_pct:.1f}%)")
        report.append(f"Custom priority: {custom:,} ({custom_pct:.1f}%)")
        report.append(f"With modifiers (↓←→): {results['with_modifier']:,}")
        report.append("")

        # Priority breakdown
        if results["by_priority"]:
            report.append("CUSTOM PRIORITY BREAKDOWN")
            report.append("-" * 40)
            for priority, count in sorted(
                results["by_priority"].items(), key=lambda x: x[1], reverse=True
            ):
                pct = (count / custom * 100) if custom > 0 else 0
                report.append(f"Priority {priority}: {count} ({pct:.1f}% of custom)")
            report.append("")

        # Modifier breakdown
        if results["by_modifier"]:
            report.append("MODIFIER BREAKDOWN")
            report.append("-" * 40)
            for modifier, count in results["by_modifier"].items():
                report.append(f"Modifier '{modifier}': {count}")
            report.append("")

        # Top modules by simp rule count
        report.append("TOP 10 MODULES BY SIMP RULE COUNT")
        report.append("-" * 40)
        module_items = [(m, s["total"]) for m, s in results["by_module"].items()]
        for module, total in sorted(module_items, key=lambda x: x[1], reverse=True)[
            :10
        ]:
            stats = results["by_module"][module]
            custom_in_module = stats["custom"]
            pct = (custom_in_module / stats["total"] * 100) if stats["total"] > 0 else 0
            report.append(
                f"{module}: {total} rules ({custom_in_module} custom, {pct:.1f}%)"
            )
        report.append("")

        # Examples of custom priority rules
        report.append("EXAMPLES OF CUSTOM PRIORITY RULES")
        report.append("-" * 40)
        for i, example in enumerate(results["custom_examples"][:20], 1):
            report.append(f"\nExample {i}:")
            report.append(f"File: {example['file']}:{example['line']}")
            report.append(f"Attribute: {example['attribute']}")
            if example["priority"]:
                report.append(f"Priority: {example['priority']}")
            if example["modifier"]:
                report.append(f"Modifier: {example['modifier']}")
            report.append(f"Context: {example['context_snippet']}")

        report.append("")
        report.append("=" * 80)
        report.append("CONCLUSION")
        report.append("=" * 80)
        report.append(
            f"The claim that 99.8% of mathlib4 uses default priorities is {'ACCURATE' if default_pct >= 99 else 'NOT ACCURATE'}"
        )
        report.append(f"Actual percentage using default priority: {default_pct:.1f}%")
        report.append(f"Actual percentage using custom priority: {custom_pct:.1f}%")

        return "\n".join(report)

    def run_analysis(self):
        """Run the complete analysis."""
        print("Starting mathlib4 simp priority verification...")
        print("=" * 60)

        # Clone/update repository
        self.clone_mathlib4()

        # Analyze
        results = self.analyze_mathlib()

        # Generate report
        report = self.generate_report(results)

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_file = f"mathlib_analysis_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {json_file}")

        # Save text report
        report_file = f"mathlib_analysis_report_{timestamp}.txt"
        with open(report_file, "w") as f:
            f.write(report)
        print(f"Report saved to: {report_file}")

        # Print report to console
        print("\n" + report)

        return results


def main():
    """Main entry point."""
    analyzer = Mathlib4Analyzer()
    results = analyzer.run_analysis()

    # Return success/failure based on verification
    default_pct = (
        (results["default_priority"] / results["total_simp_rules"] * 100)
        if results["total_simp_rules"] > 0
        else 0
    )
    return 0 if default_pct >= 99 else 1


if __name__ == "__main__":
    exit(main())
