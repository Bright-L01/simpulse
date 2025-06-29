#!/usr/bin/env python3
"""Optimize the leansat project and measure real performance improvements."""

import asyncio
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.lean_interface import LeanInterface
from src.core.performance_analyzer import PerformanceAnalyzer
from src.core.rule_extractor import RuleExtractor
from src.optimization.optimizer import Optimizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LeansatOptimizer:
    def __init__(self):
        self.project_path = Path("analyzed_repos/leansat")
        self.lean_interface = LeanInterface()
        self.rule_extractor = RuleExtractor()
        self.performance_analyzer = PerformanceAnalyzer()
        self.optimizer = Optimizer()

    async def measure_baseline_performance(self) -> Dict:
        """Measure current compilation times for leansat."""
        logger.info("Measuring baseline performance...")

        # Clean build first
        subprocess.run(["lake", "clean"], cwd=self.project_path, capture_output=True)

        # Measure full build time
        start_time = time.time()
        result = subprocess.run(
            ["lake", "build"], cwd=self.project_path, capture_output=True, text=True
        )
        build_time = time.time() - start_time

        if result.returncode != 0:
            logger.error(f"Build failed: {result.stderr}")
            return None

        # Measure individual file compilation times
        file_times = {}
        lean_files = list(self.project_path.glob("**/*.lean"))

        for lean_file in lean_files:
            if "lake-packages" in str(lean_file):
                continue

            start = time.time()
            result = subprocess.run(
                ["lean", str(lean_file)],
                cwd=self.project_path,
                capture_output=True,
                text=True,
            )
            elapsed = time.time() - start

            if result.returncode == 0:
                file_times[str(lean_file.relative_to(self.project_path))] = elapsed

        return {
            "total_build_time": build_time,
            "file_times": file_times,
            "total_files": len(file_times),
            "slowest_files": sorted(
                file_times.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

    async def extract_and_analyze_rules(self) -> Dict:
        """Extract all simp rules and analyze their usage."""
        logger.info("Extracting simp rules from leansat...")

        all_rules = []
        rule_usage = {}

        for lean_file in self.project_path.glob("**/*.lean"):
            if "lake-packages" in str(lean_file):
                continue

            try:
                module_rules = await self.rule_extractor.extract_rules_from_file(
                    lean_file
                )
                if module_rules and module_rules.rules:
                    for rule in module_rules.rules:
                        all_rules.append(
                            {
                                "name": rule.name,
                                "priority": rule.priority,
                                "file": str(lean_file.relative_to(self.project_path)),
                                "pattern": rule.pattern,
                            }
                        )

                        # Track usage patterns
                        pattern = rule.pattern or "unknown"
                        rule_usage[pattern] = rule_usage.get(pattern, 0) + 1

            except Exception as e:
                logger.warning(f"Error extracting from {lean_file}: {e}")

        return {
            "total_rules": len(all_rules),
            "rules": all_rules,
            "usage_patterns": rule_usage,
            "priority_distribution": self._analyze_priorities(all_rules),
        }

    def _analyze_priorities(self, rules: List[Dict]) -> Dict:
        """Analyze priority distribution."""
        priorities = {}
        for rule in rules:
            p = rule["priority"]
            priorities[p] = priorities.get(p, 0) + 1
        return priorities

    async def optimize_rules(self, rules_data: Dict) -> List[Dict]:
        """Generate optimized rule priorities."""
        logger.info("Generating optimized rule configuration...")

        # Group rules by pattern frequency and file location
        optimized_rules = []

        # Sort rules by usage frequency (most used = higher priority)
        rules_by_usage = {}
        for rule in rules_data["rules"]:
            pattern = rule["pattern"]
            usage_count = rules_data["usage_patterns"].get(pattern, 0)
            rule["usage_count"] = usage_count

            if pattern not in rules_by_usage:
                rules_by_usage[pattern] = []
            rules_by_usage[pattern].append(rule)

        # Assign priorities based on usage patterns
        priority = 1000  # Start with high priority
        for pattern, count in sorted(
            rules_data["usage_patterns"].items(), key=lambda x: x[1], reverse=True
        ):
            for rule in rules_by_usage.get(pattern, []):
                optimized_rule = rule.copy()
                optimized_rule["new_priority"] = priority
                optimized_rule["priority_delta"] = priority - rule["priority"]
                optimized_rules.append(optimized_rule)
            priority -= 10  # Decrease priority for less frequently used patterns

        return optimized_rules

    async def apply_optimizations(self, optimized_rules: List[Dict]) -> Path:
        """Create optimized version of the project."""
        logger.info("Applying optimizations...")

        # Create a copy of the project
        optimized_path = Path("optimized_leansat")
        if optimized_path.exists():
            shutil.rmtree(optimized_path)
        shutil.copytree(self.project_path, optimized_path)

        # Apply priority changes to each file
        changes_by_file = {}
        for rule in optimized_rules:
            if rule["priority_delta"] != 0:
                file_path = rule["file"]
                if file_path not in changes_by_file:
                    changes_by_file[file_path] = []
                changes_by_file[file_path].append(rule)

        for file_path, changes in changes_by_file.items():
            full_path = optimized_path / file_path
            if not full_path.exists():
                continue

            content = full_path.read_text()

            # Apply each priority change
            for change in changes:
                old_pattern = f"@[simp] theorem {change['name']}"
                new_pattern = (
                    f"@[simp {change['new_priority']}] theorem {change['name']}"
                )

                if change["priority"] != 1000:  # Already has custom priority
                    old_pattern = (
                        f"@[simp {change['priority']}] theorem {change['name']}"
                    )

                content = content.replace(old_pattern, new_pattern)

            full_path.write_text(content)

        return optimized_path

    async def measure_optimized_performance(self, optimized_path: Path) -> Dict:
        """Measure performance of optimized version."""
        logger.info("Measuring optimized performance...")

        # Clean build first
        subprocess.run(["lake", "clean"], cwd=optimized_path, capture_output=True)

        # Measure full build time
        start_time = time.time()
        result = subprocess.run(
            ["lake", "build"], cwd=optimized_path, capture_output=True, text=True
        )
        build_time = time.time() - start_time

        if result.returncode != 0:
            logger.error(f"Optimized build failed: {result.stderr}")
            return None

        # Measure individual file compilation times
        file_times = {}
        lean_files = list(optimized_path.glob("**/*.lean"))

        for lean_file in lean_files:
            if "lake-packages" in str(lean_file):
                continue

            start = time.time()
            result = subprocess.run(
                ["lean", str(lean_file)],
                cwd=optimized_path,
                capture_output=True,
                text=True,
            )
            elapsed = time.time() - start

            if result.returncode == 0:
                file_times[str(lean_file.relative_to(optimized_path))] = elapsed

        return {
            "total_build_time": build_time,
            "file_times": file_times,
            "total_files": len(file_times),
            "slowest_files": sorted(
                file_times.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

    async def generate_report(
        self,
        baseline: Dict,
        optimized: Dict,
        rules_data: Dict,
        optimized_rules: List[Dict],
    ) -> Dict:
        """Generate comprehensive optimization report."""
        logger.info("Generating optimization report...")

        # Calculate improvements
        build_improvement = (
            (baseline["total_build_time"] - optimized["total_build_time"])
            / baseline["total_build_time"]
            * 100
        )

        # File-by-file improvements
        file_improvements = {}
        for file, baseline_time in baseline["file_times"].items():
            if file in optimized["file_times"]:
                opt_time = optimized["file_times"][file]
                improvement = (baseline_time - opt_time) / baseline_time * 100
                file_improvements[file] = {
                    "baseline": baseline_time,
                    "optimized": opt_time,
                    "improvement": improvement,
                }

        # Count rules that were actually changed
        rules_changed = sum(1 for r in optimized_rules if r["priority_delta"] != 0)

        report = {
            "project": "leanprover/leansat",
            "optimization_summary": {
                "total_rules": rules_data["total_rules"],
                "rules_optimized": rules_changed,
                "build_time_improvement": round(build_improvement, 1),
                "baseline_build_time": round(baseline["total_build_time"], 2),
                "optimized_build_time": round(optimized["total_build_time"], 2),
            },
            "file_improvements": file_improvements,
            "top_improvements": sorted(
                file_improvements.items(),
                key=lambda x: x[1]["improvement"],
                reverse=True,
            )[:5],
            "optimization_details": {
                "priority_changes": [
                    {
                        "rule": r["name"],
                        "file": r["file"],
                        "old_priority": r["priority"],
                        "new_priority": r["new_priority"],
                        "usage_count": r["usage_count"],
                    }
                    for r in optimized_rules
                    if r["priority_delta"] != 0
                ][
                    :10
                ]  # Top 10 changes
            },
        }

        return report

    async def run_optimization(self):
        """Run the complete optimization process."""
        try:
            # Step 1: Measure baseline
            logger.info("Step 1: Measuring baseline performance...")
            baseline = await self.measure_baseline_performance()
            if not baseline:
                logger.error("Failed to measure baseline performance")
                return

            # Step 2: Extract and analyze rules
            logger.info("Step 2: Extracting and analyzing simp rules...")
            rules_data = await self.extract_and_analyze_rules()

            # Step 3: Generate optimized configuration
            logger.info("Step 3: Generating optimized rule configuration...")
            optimized_rules = await self.optimize_rules(rules_data)

            # Step 4: Apply optimizations
            logger.info("Step 4: Applying optimizations...")
            optimized_path = await self.apply_optimizations(optimized_rules)

            # Step 5: Measure optimized performance
            logger.info("Step 5: Measuring optimized performance...")
            optimized_perf = await self.measure_optimized_performance(optimized_path)
            if not optimized_perf:
                logger.error("Failed to measure optimized performance")
                return

            # Step 6: Generate report
            logger.info("Step 6: Generating optimization report...")
            report = await self.generate_report(
                baseline, optimized_perf, rules_data, optimized_rules
            )

            # Save report
            report_path = Path("leansat_optimization_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Optimization complete! Report saved to {report_path}")
            logger.info(
                f"Build time improvement: {report['optimization_summary']['build_time_improvement']}%"
            )

            return report

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise


async def main():
    optimizer = LeansatOptimizer()
    await optimizer.run_optimization()


if __name__ == "__main__":
    asyncio.run(main())
