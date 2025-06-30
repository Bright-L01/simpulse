#!/usr/bin/env python3
"""Validate Simpulse on actual mathlib4 modules with real compilation."""

import csv
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class Mathlib4Validator:
    """Validates Simpulse performance on real mathlib4 modules."""

    def __init__(self):
        self.work_dir = Path("mathlib4_validation")
        self.results_dir = Path("validation_results")
        self.results_dir.mkdir(exist_ok=True)
        self.results = []

    def setup_mathlib4(self) -> Path:
        """Clone mathlib4 if needed."""
        mathlib_path = self.work_dir / "mathlib4"

        if not mathlib_path.exists():
            print("📥 Cloning mathlib4 (this will take a few minutes)...")
            try:
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "100",
                        "https://github.com/leanprover-community/mathlib4.git",
                        str(mathlib_path),
                    ],
                    check=True,
                )
                print("✅ Clone complete")
            except subprocess.CalledProcessError as e:
                print(f"❌ Clone failed: {e}")
                return None
        else:
            print("✅ Using existing mathlib4 clone")
            # Update to latest
            subprocess.run(["git", "pull"], cwd=mathlib_path, capture_output=True)

        return mathlib_path

    def select_representative_modules(self) -> List[Dict[str, str]]:
        """Select representative modules for testing."""
        modules = [
            {
                "name": "Algebra.Group.Basic",
                "path": "Mathlib/Algebra/Group/Basic.lean",
                "description": "Basic group theory",
            },
            {
                "name": "Data.List.Basic",
                "path": "Mathlib/Data/List/Basic.lean",
                "description": "List operations",
            },
            {
                "name": "Data.Nat.Basic",
                "path": "Mathlib/Data/Nat/Basic.lean",
                "description": "Natural number properties",
            },
            {
                "name": "Logic.Basic",
                "path": "Mathlib/Logic/Basic.lean",
                "description": "Basic logic theorems",
            },
            {
                "name": "Order.Basic",
                "path": "Mathlib/Order/Basic.lean",
                "description": "Order relations",
            },
        ]
        return modules

    def extract_simp_rules(self, file_path: Path) -> List[Dict]:
        """Extract simp rules from a Lean file."""
        rules = []

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Pattern to match simp attributes and theorem names
            simp_pattern = re.compile(
                r"@\[([^\]]*simp[^\]]*)\]\s*(?:theorem|lemma|def)\s+(\w+)", re.MULTILINE
            )

            for match in simp_pattern.finditer(content):
                attr_text = match.group(1)
                theorem_name = match.group(2)

                # Extract priority if specified
                priority_match = re.search(r"simp\s+(\d+)", attr_text)
                priority = int(priority_match.group(1)) if priority_match else 1000

                # Classify rule by pattern
                rule_type = self._classify_rule(
                    theorem_name, content[match.start() : match.end() + 200]
                )

                rules.append(
                    {
                        "name": theorem_name,
                        "priority": priority,
                        "type": rule_type,
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )

        except Exception as e:
            print(f"Error extracting rules from {file_path}: {e}")

        return rules

    def _classify_rule(self, name: str, context: str) -> str:
        """Classify a simp rule by its likely usage frequency."""
        name_lower = name.lower()

        # High frequency patterns
        if any(
            p in name_lower
            for p in ["add_zero", "zero_add", "mul_one", "one_mul", "mul_zero"]
        ):
            return "arithmetic_basic"
        elif any(p in name_lower for p in ["append_nil", "nil_append", "length_nil"]):
            return "list_basic"
        elif any(
            p in name_lower for p in ["and_true", "true_and", "or_false", "false_or"]
        ):
            return "bool_basic"

        # Medium frequency
        elif "comm" in name_lower or "assoc" in name_lower:
            return "algebraic"
        elif "distrib" in name_lower:
            return "distributive"

        # Low frequency
        elif "iff" in name_lower or "equiv" in name_lower:
            return "equivalence"
        elif len(name) > 20 or "aux" in name_lower:
            return "complex"

        return "general"

    def optimize_priorities(self, rules: List[Dict]) -> List[Dict]:
        """Assign optimized priorities based on rule classification."""
        # Priority ranges by type
        priority_map = {
            "arithmetic_basic": 2000,  # Highest
            "list_basic": 1900,
            "bool_basic": 1800,
            "algebraic": 1500,
            "distributive": 1400,
            "general": 1000,  # Default
            "equivalence": 700,
            "complex": 500,  # Lowest
        }

        optimized = []
        for rule in rules:
            opt_rule = rule.copy()
            opt_rule["new_priority"] = priority_map.get(rule["type"], 1000)
            opt_rule["priority_change"] = opt_rule["new_priority"] - rule["priority"]
            optimized.append(opt_rule)

        return optimized

    def apply_optimization(self, file_path: Path, optimized_rules: List[Dict]) -> Path:
        """Apply optimized priorities to create new file."""
        content = file_path.read_text(encoding="utf-8", errors="ignore")

        # Create optimized version
        for rule in optimized_rules:
            if rule["priority_change"] != 0:
                # Replace @[simp] with @[simp PRIORITY]
                old_pattern = f"@\\[([^\\]]*simp)([^\\]]*)\\]\\s*(?:theorem|lemma|def)\\s+{rule['name']}"
                new_attr = f"@[\\1 {rule['new_priority']}\\2] theorem {rule['name']}"

                content = re.sub(old_pattern, new_attr, content)

        # Save optimized version
        opt_path = file_path.parent / f"{file_path.stem}_optimized{file_path.suffix}"
        opt_path.write_text(content)

        return opt_path

    def measure_compilation_time(
        self, mathlib_path: Path, module_path: str, label: str, timeout: int = 300
    ) -> Optional[float]:
        """Measure compilation time for a module."""
        print(f"\n⏱️  Measuring {label} compilation time...")

        # Clean build cache
        subprocess.run(["lake", "clean"], cwd=mathlib_path, capture_output=True)

        # Run compilation with timing
        start_time = time.time()

        try:
            result = subprocess.run(
                ["lake", "build", module_path.replace(".lean", "")],
                cwd=mathlib_path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                print(f"✅ Compilation successful: {elapsed:.2f}s")
                return elapsed
            else:
                print(f"❌ Compilation failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print(f"❌ Compilation timeout after {timeout}s")
            return None
        except Exception as e:
            print(f"❌ Compilation error: {e}")
            return None

    def validate_module(self, mathlib_path: Path, module: Dict) -> Dict:
        """Validate a single module."""
        print(f"\n{'='*70}")
        print(f"📦 Validating: {module['name']}")
        print(f"📄 Description: {module['description']}")
        print(f"{'='*70}")

        result = {
            "module": module["name"],
            "description": module["description"],
            "baseline_time": None,
            "optimized_time": None,
            "improvement_percent": None,
            "rules_analyzed": 0,
            "rules_optimized": 0,
            "status": "pending",
        }

        try:
            # Extract simp rules
            file_path = mathlib_path / module["path"]
            if not file_path.exists():
                print(f"❌ Module file not found: {file_path}")
                result["status"] = "file_not_found"
                return result

            rules = self.extract_simp_rules(file_path)
            result["rules_analyzed"] = len(rules)

            print(f"\n📊 Found {len(rules)} simp rules")

            if len(rules) == 0:
                print("⚠️  No simp rules found, skipping optimization")
                result["status"] = "no_simp_rules"
                return result

            # Show rule distribution
            type_counts = {}
            for rule in rules:
                type_counts[rule["type"]] = type_counts.get(rule["type"], 0) + 1

            print("\n📈 Rule distribution:")
            for rule_type, count in sorted(type_counts.items()):
                print(f"   {rule_type}: {count} rules")

            # Measure baseline
            baseline_time = self.measure_compilation_time(
                mathlib_path, module["path"], "baseline"
            )

            if baseline_time is None:
                result["status"] = "baseline_failed"
                return result

            result["baseline_time"] = baseline_time

            # Optimize priorities
            optimized_rules = self.optimize_priorities(rules)
            rules_changed = sum(1 for r in optimized_rules if r["priority_change"] != 0)
            result["rules_optimized"] = rules_changed

            print(f"\n🔧 Optimizing {rules_changed} rules...")

            if rules_changed == 0:
                print("ℹ️  No optimization needed (already optimal)")
                result["status"] = "already_optimal"
                result["optimized_time"] = baseline_time
                result["improvement_percent"] = 0
                return result

            # Apply optimization
            opt_file = self.apply_optimization(file_path, optimized_rules)

            # Replace original with optimized for testing
            shutil.copy(opt_file, file_path)

            # Measure optimized
            optimized_time = self.measure_compilation_time(
                mathlib_path, module["path"], "optimized"
            )

            # Restore original
            subprocess.run(["git", "checkout", module["path"]], cwd=mathlib_path)
            opt_file.unlink()

            if optimized_time is None:
                result["status"] = "optimized_failed"
                return result

            result["optimized_time"] = optimized_time
            result["improvement_percent"] = (
                (baseline_time - optimized_time) / baseline_time * 100
            )
            result["status"] = "success"

            print("\n🏆 Results:")
            print(f"   Baseline: {baseline_time:.2f}s")
            print(f"   Optimized: {optimized_time:.2f}s")
            print(f"   Improvement: {result['improvement_percent']:.1f}%")

        except Exception as e:
            print(f"❌ Validation error: {e}")
            result["status"] = "error"
            result["error"] = str(e)

        return result

    def run_validation(self):
        """Run complete validation."""
        print("🚀 MATHLIB4 VALIDATION SUITE")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Setup mathlib4
        mathlib_path = self.setup_mathlib4()
        if not mathlib_path:
            return

        # Select modules
        modules = self.select_representative_modules()

        # Validate each module
        for module in modules:
            result = self.validate_module(mathlib_path, module)
            self.results.append(result)

        # Generate report
        self.generate_report()

    def generate_report(self):
        """Generate validation report."""
        print("\n" + "=" * 70)
        print("📊 VALIDATION SUMMARY")
        print("=" * 70)

        # Save CSV
        csv_path = (
            self.results_dir
            / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "module",
                    "description",
                    "status",
                    "rules_analyzed",
                    "rules_optimized",
                    "baseline_time",
                    "optimized_time",
                    "improvement_percent",
                ],
            )
            writer.writeheader()
            writer.writerows(self.results)

        print(f"\n📄 Results saved to: {csv_path}")

        # Summary statistics
        successful = [r for r in self.results if r["status"] == "success"]

        if successful:
            avg_improvement = sum(r["improvement_percent"] for r in successful) / len(
                successful
            )
            total_rules = sum(r["rules_analyzed"] for r in self.results)
            total_optimized = sum(r["rules_optimized"] for r in self.results)

            print(f"\n✅ Successful validations: {len(successful)}/{len(self.results)}")
            print(f"📊 Average improvement: {avg_improvement:.1f}%")
            print(f"📏 Total rules analyzed: {total_rules}")
            print(f"🔧 Total rules optimized: {total_optimized}")

            print("\n📈 Module Results:")
            for r in successful:
                print(f"   {r['module']}: {r['improvement_percent']:.1f}% improvement")
        else:
            print("\n❌ No successful validations")

        # Create markdown report
        self.create_markdown_report()

    def create_markdown_report(self):
        """Create detailed markdown report."""
        report_path = self.results_dir / "validation_report.md"

        content = f"""# Mathlib4 Validation Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Value |
|--------|-------|
| Modules tested | {len(self.results)} |
| Successful | {sum(1 for r in self.results if r['status'] == 'success')} |
| Total rules analyzed | {sum(r['rules_analyzed'] for r in self.results)} |
| Rules optimized | {sum(r['rules_optimized'] for r in self.results)} |

## Results by Module

| Module | Rules | Optimized | Baseline (s) | Optimized (s) | Improvement |
|--------|-------|-----------|--------------|---------------|-------------|
"""

        for r in self.results:
            baseline = f"{r['baseline_time']:.2f}" if r["baseline_time"] else "N/A"
            optimized = f"{r['optimized_time']:.2f}" if r["optimized_time"] else "N/A"
            improvement = (
                f"{r['improvement_percent']:.1f}%"
                if r["improvement_percent"] is not None
                else "N/A"
            )

            content += f"| {r['module']} | {r['rules_analyzed']} | {r['rules_optimized']} | {baseline} | {optimized} | {improvement} |\n"

        content += """
## Validation Status

"""
        for r in self.results:
            content += f"- **{r['module']}**: {r['status']}\n"

        report_path.write_text(content)
        print(f"📝 Markdown report: {report_path}")


def main():
    """Run mathlib4 validation."""
    validator = Mathlib4Validator()
    validator.run_validation()


if __name__ == "__main__":
    main()
