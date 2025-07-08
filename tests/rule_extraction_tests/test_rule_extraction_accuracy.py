"""
Test suite for rule extraction accuracy on real mathlib4 files.

This test file evaluates the REAL capability of Simpulse - extracting simp rules
from Lean 4 files. It tests on actual mathlib4 code and documents exact accuracy.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from simpulse.evolution.models import SimpRule
from simpulse.evolution.rule_extractor import RuleExtractor


class RuleExtractionTester:
    """Test rule extraction accuracy on real mathlib4 files."""

    def __init__(self):
        self.extractor = RuleExtractor()
        self.results = []

    def test_file(self, file_path: Path) -> Dict[str, Any]:
        """Test rule extraction on a single file."""
        print(f"\n{'='*60}")
        print(f"Testing: {file_path.name}")
        print(f"{'='*60}")

        # Read file content
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return {"file": str(file_path), "error": str(e), "success": False}

        # Count actual simp rules manually
        actual_simp_count = content.count("@[simp")
        print(f"Manual count of '@[simp': {actual_simp_count}")

        # Extract rules using our extractor
        module_rules = self.extractor.extract_rules_from_file(file_path)
        extracted_count = len(module_rules.rules)
        print(f"Extracted rules: {extracted_count}")

        # Analyze results
        accuracy = (extracted_count / actual_simp_count * 100) if actual_simp_count > 0 else 100

        result = {
            "file": str(file_path),
            "file_size_bytes": len(content),
            "line_count": content.count("\n") + 1,
            "actual_simp_count": actual_simp_count,
            "extracted_count": extracted_count,
            "accuracy_percent": round(accuracy, 2),
            "success": True,
            "rules": [],
        }

        # Show sample extracted rules
        print(f"\nSample extracted rules (first 5):")
        for i, rule in enumerate(module_rules.rules[:5]):
            print(f"  {i+1}. {rule.name} (priority: {rule.priority})")
            result["rules"].append(
                {"name": rule.name, "priority": str(rule.priority), "line": rule.location.line}
            )

        if extracted_count > 5:
            print(f"  ... and {extracted_count - 5} more")

        # Find missed rules
        if extracted_count < actual_simp_count:
            print(f"\n⚠️  Missed {actual_simp_count - extracted_count} rules")
            self._analyze_missed_rules(content, module_rules.rules)
        elif extracted_count > actual_simp_count:
            print(f"\n⚠️  Over-extracted by {extracted_count - actual_simp_count} rules")

        return result

    def _analyze_missed_rules(self, content: str, extracted_rules: List[SimpRule]):
        """Analyze which simp rules were missed."""
        lines = content.split("\n")
        extracted_lines = {rule.location.line for rule in extracted_rules}

        print("\nMissed simp attributes:")
        for i, line in enumerate(lines):
            if "@[simp" in line and (i + 1) not in extracted_lines:
                # Show context
                start = max(0, i - 1)
                end = min(len(lines), i + 3)
                print(f"\n  Line {i+1}:")
                for j in range(start, end):
                    prefix = ">>> " if j == i else "    "
                    print(f"  {prefix}{lines[j]}")

    def test_edge_cases(self):
        """Test specific edge cases for rule extraction."""
        print("\n" + "=" * 60)
        print("EDGE CASE TESTING")
        print("=" * 60)

        edge_cases = [
            # Multi-line simp attribute
            """
@[simp]
theorem multi_line : 1 = 1 := rfl
            """,
            # Simp with priority
            """
@[simp high]
theorem high_priority : 2 = 2 := rfl

@[simp low]  
theorem low_priority : 3 = 3 := rfl

@[simp 1000]
theorem numeric_priority : 4 = 4 := rfl
            """,
            # Simp with direction
            """
@[simp ←]
theorem backward_simp : a = b ↔ b = a := sorry

@[simp ↓]
theorem downward_simp : true = true := rfl
            """,
            # Complex attribute combinations
            """
@[simp high ←]
theorem complex_attr : x + y = y + x := sorry

@[inline, simp]
theorem multi_attr : 0 + n = n := sorry
            """,
            # Simp on different declaration types
            """
@[simp]
def simp_def (n : Nat) : Nat := n + 0

@[simp]
instance : Inhabited Nat := ⟨0⟩

@[simp]
axiom simp_axiom : ∀ n : Nat, n = n
            """,
            # Edge cases that might break regex
            """
-- This is not a simp rule: @[simp]
/- Block comment with @[simp] inside -/

@[simp] -- inline comment
theorem with_comment : true := trivial

theorem not_simp : false := sorry  -- @[simp] in comment

@[simps] -- similar but different attribute
def not_simp_attr : Nat → Nat := id
            """,
        ]

        for i, test_case in enumerate(edge_cases):
            print(f"\nEdge case {i+1}:")
            print("Code:")
            print(test_case.strip())

            # Create temporary file
            temp_file = Path(f"/tmp/edge_case_{i}.lean")
            temp_file.write_text(test_case)

            # Extract rules
            module_rules = self.extractor.extract_rules_from_file(temp_file)

            # Count expected
            expected = len(
                [
                    line
                    for line in test_case.split("\n")
                    if line.strip().startswith("@[simp")
                    and not line.strip().startswith("--")
                    and "@[simps]" not in line
                ]
            )

            print(f"Expected: {expected}, Extracted: {len(module_rules.rules)}")
            if len(module_rules.rules) != expected:
                print("❌ FAILED")
            else:
                print("✅ PASSED")

            # Clean up
            temp_file.unlink()

    def run_all_tests(self, test_files: List[Path]):
        """Run tests on all provided files."""
        print("RULE EXTRACTION ACCURACY TEST")
        print("Testing on real mathlib4 files")

        total_actual = 0
        total_extracted = 0

        for file_path in test_files:
            result = self.test_file(file_path)
            self.results.append(result)

            if result["success"]:
                total_actual += result["actual_simp_count"]
                total_extracted += result["extracted_count"]

        # Overall statistics
        print(f"\n{'='*60}")
        print("OVERALL RESULTS")
        print(f"{'='*60}")
        print(f"Files tested: {len(test_files)}")
        print(f"Total actual simp rules: {total_actual}")
        print(f"Total extracted rules: {total_extracted}")
        overall_accuracy = (total_extracted / total_actual * 100) if total_actual > 0 else 100
        print(f"Overall accuracy: {overall_accuracy:.2f}%")

        # Save results
        self._save_results()

        # Run edge case tests
        self.test_edge_cases()

        return self.results

    def _save_results(self):
        """Save test results to JSON."""
        output_file = Path("tests/rule_extraction_tests/extraction_results.json")
        output_file.parent.mkdir(exist_ok=True)

        result_data = {
            "test_date": "2025-07-04",
            "results": self.results,
            "summary": {
                "files_tested": len(self.results),
                "total_actual_rules": sum(
                    r["actual_simp_count"] for r in self.results if r["success"]
                ),
                "total_extracted_rules": sum(
                    r["extracted_count"] for r in self.results if r["success"]
                ),
                "overall_accuracy": (
                    round(
                        sum(r["extracted_count"] for r in self.results if r["success"])
                        / sum(r["actual_simp_count"] for r in self.results if r["success"])
                        * 100,
                        2,
                    )
                    if any(r["success"] for r in self.results)
                    else 0
                ),
            },
        }

        with open(output_file, "w") as f:
            json.dump(result_data, f, indent=2)

        print(f"\nResults saved to: {output_file}")


def main():
    """Run rule extraction tests on mathlib4 files."""
    # Use the mathlib4 files we already have
    test_files = [
        Path("mathlib4_test_modules/Nat_Basic.lean"),
        Path("mathlib4_test_modules/List_Basic.lean"),
        Path("mathlib4_test_modules/Logic_Basic.lean"),
        Path("mathlib4_test_modules/Group_Basic.lean"),
        Path("mathlib4_test_modules/Order_Basic.lean"),
    ]

    # Add more test files if available
    mathlib_path = Path("/path/to/mathlib4")  # Update if you have local mathlib4
    if mathlib_path.exists():
        # Add some core mathlib4 files
        additional_files = [
            mathlib_path / "Mathlib/Data/Nat/Basic.lean",
            mathlib_path / "Mathlib/Data/List/Basic.lean",
            mathlib_path / "Mathlib/Logic/Basic.lean",
            mathlib_path / "Mathlib/Algebra/Group/Basic.lean",
            mathlib_path / "Mathlib/Order/Basic.lean",
        ]
        test_files.extend([f for f in additional_files if f.exists()])

    # Run tests
    tester = RuleExtractionTester()
    results = tester.run_all_tests(test_files)

    return results


if __name__ == "__main__":
    main()
