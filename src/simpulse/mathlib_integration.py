"""
Real mathlib4 integration for Simpulse optimization.

Provides specialized support for working with mathlib4 projects, including
common optimization patterns and mathlib-specific simp rule analysis.
"""

import time
from pathlib import Path
from typing import Any, Optional

from .errors import ErrorCategory, ErrorContext, ErrorHandler, ErrorSeverity
from .monitoring import OptimizationMetrics, PerformanceMonitor
from .optimization.optimizer import OptimizationResult, SimpOptimizer


class MathlibIntegration:
    """Integration layer for mathlib4 projects."""

    # Common mathlib4 modules that benefit from optimization
    PRIORITY_MODULES = [
        "Mathlib.Data.List.Basic",
        "Mathlib.Data.Nat.Basic",
        "Mathlib.Data.Int.Basic",
        "Mathlib.Algebra.Ring.Basic",
        "Mathlib.Algebra.Group.Basic",
        "Mathlib.Logic.Basic",
        "Mathlib.Order.Basic",
        "Mathlib.Tactic.Simp.Basic",
    ]

    # Optimization strategies best suited for different mathlib areas
    MODULE_STRATEGIES = {
        "Data": "frequency",  # Data structures benefit from frequency-based optimization
        "Algebra": "performance",  # Algebra benefits from performance optimization
        "Logic": "balanced",  # Logic benefits from balanced approach
        "Tactic": "conservative",  # Tactics need conservative optimization
        "Order": "complexity",  # Order theory benefits from complexity-based
        "Category": "context_aware",  # Category theory needs context awareness
    }

    def __init__(self):
        self.error_handler = ErrorHandler()
        self.monitor = PerformanceMonitor()

    def detect_mathlib_project(self, project_path: Path) -> dict[str, Any]:
        """Detect if a project uses mathlib4 and gather information."""

        info = {
            "is_mathlib_project": False,
            "mathlib_version": None,
            "lakefile_path": None,
            "modules_found": [],
            "dependencies": [],
            "optimization_potential": "unknown",
        }

        try:
            # Check for lakefile.lean
            lakefile = project_path / "lakefile.lean"
            if lakefile.exists():
                info["lakefile_path"] = str(lakefile)

                content = lakefile.read_text()

                # Check for mathlib dependency
                if "mathlib" in content.lower():
                    info["is_mathlib_project"] = True

                    # Try to extract mathlib version/commit
                    lines = content.split("\n")
                    for line in lines:
                        if "mathlib" in line.lower() and ("git" in line or "version" in line):
                            info["mathlib_version"] = line.strip()
                            break

            # Scan for mathlib modules
            mathlib_modules = []
            for lean_file in project_path.glob("**/*.lean"):
                if "lake-packages" in str(lean_file):
                    continue

                try:
                    content = lean_file.read_text()
                    for line in content.split("\n"):
                        if line.strip().startswith("import") and "Mathlib" in line:
                            module = line.replace("import", "").strip()
                            mathlib_modules.append(module)
                except Exception:
                    continue

            info["modules_found"] = list(set(mathlib_modules))

            # Assess optimization potential
            if len(mathlib_modules) > 10:
                info["optimization_potential"] = "high"
            elif len(mathlib_modules) > 3:
                info["optimization_potential"] = "medium"
            elif len(mathlib_modules) > 0:
                info["optimization_potential"] = "low"

        except Exception as e:
            self.error_handler.handle_error(
                category=ErrorCategory.FILE_ACCESS,
                severity=ErrorSeverity.MEDIUM,
                message="Failed to detect mathlib project",
                context=ErrorContext(operation="detect_mathlib", file_path=project_path),
                exception=e,
            )

        return info

    def suggest_optimization_strategy(self, project_info: dict[str, Any]) -> str:
        """Suggest the best optimization strategy for a mathlib project."""

        if not project_info.get("is_mathlib_project", False):
            return "balanced"  # Default for non-mathlib projects

        modules = project_info.get("modules_found", [])

        # Count modules by area
        area_counts = {}
        for module in modules:
            for area, strategy in self.MODULE_STRATEGIES.items():
                if area in module:
                    area_counts[area] = area_counts.get(area, 0) + 1

        if not area_counts:
            return "balanced"

        # Return strategy for most common area
        dominant_area = max(area_counts.items(), key=lambda x: x[1])[0]
        return self.MODULE_STRATEGIES.get(dominant_area, "balanced")

    def optimize_mathlib_project(
        self, project_path: Path, strategy: Optional[str] = None, max_rules: int = 100
    ) -> dict[str, Any]:
        """Optimize a mathlib4 project with specialized handling."""

        # Detect project characteristics
        project_info = self.detect_mathlib_project(project_path)

        if not project_info["is_mathlib_project"]:
            return {
                "error": "Project does not appear to use mathlib4",
                "suggestion": "Use regular optimization for non-mathlib projects",
            }

        # Choose strategy
        if strategy is None:
            strategy = self.suggest_optimization_strategy(project_info)

        print(f"🧮 Mathlib4 project detected:")
        print(f"   📚 Modules found: {len(project_info['modules_found'])}")
        print(f"   🎯 Optimization potential: {project_info['optimization_potential']}")
        print(f"   🔧 Using strategy: {strategy}")

        try:
            # Run optimization with mathlib-specific settings
            optimizer = SimpOptimizer(strategy=strategy)

            # Analyze project
            analysis = optimizer.analyze(project_path)

            # Limit rules for mathlib projects (they can be huge)
            if len(analysis["rules"]) > max_rules:
                print(f"   ⚠️  Large project: limiting to {max_rules} rules")
                analysis["rules"] = analysis["rules"][:max_rules]

            # Generate optimizations
            optimization_result = optimizer.optimize(analysis)

            # Record metrics for mathlib-specific monitoring
            metrics = OptimizationMetrics(
                timestamp=time.time(),
                strategy=strategy,
                project_path=str(project_path),
                rules_analyzed=len(analysis["rules"]),
                optimizations_applied=len(optimization_result.changes),
                estimated_speedup=optimization_result.estimated_improvement,
                metadata={
                    "mathlib_modules": len(project_info["modules_found"]),
                    "optimization_potential": project_info["optimization_potential"],
                    "is_mathlib_project": True,
                },
            )

            self.monitor.record_optimization(metrics)

            result = {
                "project_info": project_info,
                "optimization_result": optimization_result,
                "strategy_used": strategy,
                "mathlib_specific_insights": self._generate_mathlib_insights(
                    project_info, optimization_result
                ),
            }

            return result

        except Exception as e:
            self.error_handler.handle_error(
                category=ErrorCategory.OPTIMIZATION,
                severity=ErrorSeverity.HIGH,
                message="Mathlib optimization failed",
                context=ErrorContext(
                    operation="optimize_mathlib_project", file_path=project_path, strategy=strategy
                ),
                exception=e,
            )

            return {"error": f"Optimization failed: {str(e)}"}

    def _generate_mathlib_insights(
        self, project_info: dict[str, Any], optimization_result: OptimizationResult
    ) -> dict[str, Any]:
        """Generate mathlib-specific optimization insights."""

        insights = {
            "module_coverage": {},
            "optimization_recommendations": [],
            "potential_issues": [],
            "performance_estimates": {},
        }

        modules = project_info.get("modules_found", [])

        # Analyze module coverage
        for area in self.MODULE_STRATEGIES.keys():
            matching_modules = [m for m in modules if area in m]
            insights["module_coverage"][area] = len(matching_modules)

        # Generate recommendations
        if len(optimization_result.changes) == 0:
            insights["optimization_recommendations"].append(
                "No optimizations found - project may already be well-optimized"
            )
        elif len(optimization_result.changes) > 50:
            insights["optimization_recommendations"].append(
                "Large number of optimizations - consider applying incrementally"
            )

        # Check for potential issues
        if (
            project_info["optimization_potential"] == "high"
            and len(optimization_result.changes) < 5
        ):
            insights["potential_issues"].append(
                "High optimization potential but few changes found - check rule extraction"
            )

        # Estimate performance impact by module area
        changes_by_area = {}
        for change in optimization_result.changes:
            file_path = change.file_path
            for area in self.MODULE_STRATEGIES.keys():
                if area.lower() in file_path.lower():
                    changes_by_area[area] = changes_by_area.get(area, 0) + 1

        insights["performance_estimates"] = changes_by_area

        return insights

    def create_mathlib_test_project(self, output_path: Path) -> bool:
        """Create a realistic mathlib4 test project for integration testing."""

        try:
            output_path.mkdir(parents=True, exist_ok=True)

            # Create lakefile.lean with mathlib dependency
            lakefile = output_path / "lakefile.lean"
            lakefile.write_text(
                """
import Lake
open Lake DSL

package «mathlib_test» where
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`pp.proofs.withType, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib «MathlibTest» where
"""
            )

            # Create lean-toolchain
            toolchain = output_path / "lean-toolchain"
            toolchain.write_text("leanprover/lean4:nightly-2024-02-01")

            # Create source directory
            src_dir = output_path / "MathlibTest"
            src_dir.mkdir(exist_ok=True)

            # Create realistic mathlib-based file
            main_file = src_dir / "Main.lean"
            main_file.write_text(
                """
-- Realistic mathlib4 test file with common patterns

import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Logic.Basic

-- Basic simp rules that commonly appear in mathlib projects

@[simp] theorem list_singleton_length (a : α) : [a].length = 1 := by rfl

@[simp] theorem nat_add_zero_eq (n : Nat) : n + 0 = n := by rfl

@[simp] theorem nat_zero_add_eq (n : Nat) : 0 + n = n := by rfl

@[simp] theorem bool_and_true (b : Bool) : b && true = b := by
  cases b <;> rfl

@[simp] theorem bool_true_and (b : Bool) : true && b = b := by
  cases b <;> rfl

-- More complex theorem that would benefit from the above simp rules
theorem example_theorem (xs : List Nat) (n : Nat) : 
  (xs ++ [n]).length = xs.length + 1 := by
  simp [List.length_append, list_singleton_length]

-- Ring-like operations
@[simp] theorem add_neg_cancel (a : ℤ) : a + (-a) = 0 := by ring

-- Logic simplifications  
@[simp] theorem and_true_iff (p : Prop) : (p ∧ True) ↔ p := by simp

-- Performance-critical rules that should have high priority
@[simp, high_priority] theorem list_mem_singleton (a b : α) : a ∈ [b] ↔ a = b := by
  simp [List.mem_singleton]
"""
            )

            # Create additional module showing different mathlib areas
            algebra_file = src_dir / "AlgebraTest.lean"
            algebra_file.write_text(
                """
-- Algebra-focused module for testing

import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Ring.Basic

-- Group theory simp rules
@[simp] theorem mul_one_eq (a : G) [Group G] : a * 1 = a := by simp

@[simp] theorem one_mul_eq (a : G) [Group G] : 1 * a = a := by simp

-- Ring theory simp rules  
@[simp] theorem zero_mul_eq (a : R) [Ring R] : 0 * a = 0 := by simp

@[simp] theorem mul_zero_eq (a : R) [Ring R] : a * 0 = 0 := by simp

-- Complex theorem using the above
theorem ring_example (a b : R) [Ring R] : a * 0 + 0 * b = 0 := by simp
"""
            )

            print(f"✅ Created mathlib4 test project at: {output_path}")
            return True

        except Exception as e:
            self.error_handler.handle_error(
                category=ErrorCategory.FILE_ACCESS,
                severity=ErrorSeverity.HIGH,
                message="Failed to create mathlib test project",
                context=ErrorContext(operation="create_test_project", file_path=output_path),
                exception=e,
            )
            return False

    def benchmark_mathlib_optimization(self, project_path: Path) -> dict[str, Any]:
        """Benchmark optimization effectiveness on a mathlib project."""

        results = {}

        # Test different strategies on the same project
        strategies = ["balanced", "performance", "frequency", "conservative"]

        for strategy in strategies:
            print(f"📊 Benchmarking strategy: {strategy}")

            try:
                result = self.optimize_mathlib_project(
                    project_path=project_path,
                    strategy=strategy,
                    max_rules=50,  # Limit for benchmarking
                )

                if "error" not in result:
                    optimization = result["optimization_result"]
                    results[strategy] = {
                        "changes": len(optimization.changes),
                        "estimated_improvement": optimization.estimated_improvement,
                        "insights": result["mathlib_specific_insights"],
                    }
                else:
                    results[strategy] = {"error": result["error"]}

            except Exception as e:
                results[strategy] = {"error": str(e)}

        # Determine best strategy
        best_strategy = None
        best_score = 0

        for strategy, result in results.items():
            if "error" not in result:
                score = result["estimated_improvement"]
                if score > best_score:
                    best_score = score
                    best_strategy = strategy

        return {
            "strategy_results": results,
            "best_strategy": best_strategy,
            "best_improvement": best_score,
            "recommendations": self._generate_benchmark_recommendations(results),
        }

    def _generate_benchmark_recommendations(self, results: dict[str, Any]) -> list[str]:
        """Generate recommendations based on benchmark results."""

        recommendations = []

        # Find strategies that worked
        working_strategies = [s for s, r in results.items() if "error" not in r]

        if not working_strategies:
            recommendations.append("All strategies failed - check project setup and dependencies")
            return recommendations

        # Find best performing strategy
        best_result = max(
            (r for r in results.values() if "error" not in r),
            key=lambda x: x.get("estimated_improvement", 0),
            default=None,
        )

        if best_result:
            best_strategy = None
            for strategy, result in results.items():
                if result == best_result:
                    best_strategy = strategy
                    break

            if best_strategy:
                recommendations.append(
                    f"Use '{best_strategy}' strategy for best results "
                    f"({best_result['estimated_improvement']}% improvement)"
                )

        # Check for consistent performers
        consistent_strategies = [
            s
            for s, r in results.items()
            if "error" not in r and r.get("estimated_improvement", 0) > 10
        ]

        if len(consistent_strategies) > 1:
            recommendations.append(
                f"Multiple effective strategies available: {', '.join(consistent_strategies)}"
            )

        return recommendations


# Convenience function for easy mathlib integration
def optimize_mathlib_project(project_path: Path, strategy: Optional[str] = None) -> dict[str, Any]:
    """Convenience function to optimize a mathlib4 project."""
    integration = MathlibIntegration()
    return integration.optimize_mathlib_project(project_path, strategy)
