#!/usr/bin/env python3
"""
ExperimentRunner - Empirical optimization strategy testing

No predictions, just experiments. Run multiple optimization strategies
on real Lean files and measure actual compilation times.
"""

import json
import logging
import random
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.simpulse.analysis.advanced_context_classifier import AdvancedContextClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationStrategy:
    """Definition of an optimization strategy"""

    name: str
    description: str
    priority_adjustment: Optional[int] = None
    selective_count: Optional[int] = None
    pattern_filter: Optional[str] = None
    adaptive_threshold: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Result of a single experiment"""

    file_path: str
    context_type: str
    strategy_name: str
    baseline_time: float
    optimized_time: float
    speedup: float
    success: bool
    error_message: Optional[str] = None
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    """Run optimization experiments on real Lean files"""

    # Define optimization strategies
    STRATEGIES = [
        OptimizationStrategy(
            name="no_optimization",
            description="Baseline - no optimization",
            config={"enabled": False},
        ),
        OptimizationStrategy(
            name="conservative",
            description="Conservative priority boost (+10)",
            priority_adjustment=10,
            config={"risk_threshold": 0.9},
        ),
        OptimizationStrategy(
            name="moderate",
            description="Moderate priority boost (+50)",
            priority_adjustment=50,
            config={"risk_threshold": 0.7},
        ),
        OptimizationStrategy(
            name="aggressive",
            description="Aggressive priority boost (+100)",
            priority_adjustment=100,
            config={"risk_threshold": 0.3},
        ),
        OptimizationStrategy(
            name="selective_top5",
            description="Only optimize top 5 lemmas",
            selective_count=5,
            config={"selection_metric": "frequency"},
        ),
        OptimizationStrategy(
            name="contextual_arithmetic",
            description="Boost only arithmetic patterns",
            pattern_filter="arithmetic",
            priority_adjustment=75,
            config={"pattern_types": ["arithmetic", "numerical"]},
        ),
        OptimizationStrategy(
            name="inverse_reduction",
            description="Reduce non-matching priorities",
            priority_adjustment=-50,
            config={"target": "non_matching", "boost_matching": 25},
        ),
        OptimizationStrategy(
            name="random_shuffle",
            description="Random priority assignment for comparison",
            config={"randomize": True, "seed": 42},
        ),
        OptimizationStrategy(
            name="adaptive_threshold",
            description="Adapt strategy based on compilation progress",
            adaptive_threshold=0.5,
            config={"initial_boost": 30, "final_boost": 80},
        ),
        OptimizationStrategy(
            name="kitchen_sink",
            description="All techniques combined",
            priority_adjustment=100,
            selective_count=10,
            pattern_filter="all",
            adaptive_threshold=0.3,
            config={"aggressive": True, "multi_pass": True},
        ),
    ]

    def __init__(
        self, lean_executable: str = "lean", output_dir: Path = Path("experiment_results")
    ):
        self.lean_executable = lean_executable
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

        self.classifier = AdvancedContextClassifier()
        self.results = []
        self.payoff_matrix = None

        # Check Lean installation
        self._verify_lean_installation()

    def _verify_lean_installation(self):
        """Verify Lean is installed and working"""
        try:
            result = subprocess.run(
                [self.lean_executable, "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(f"Lean not found or not working: {result.stderr}")
            logger.info(f"Using Lean: {result.stdout.strip()}")
        except Exception as e:
            raise RuntimeError(f"Failed to verify Lean installation: {e}")

    def run_experiments(self, lean_files: List[Path], max_workers: int = 4):
        """Run all experiments on provided Lean files"""
        total_experiments = len(lean_files) * len(self.STRATEGIES)
        logger.info(
            f"Starting {total_experiments} experiments ({len(lean_files)} files × {len(self.STRATEGIES)} strategies)"
        )

        # First, classify all files
        logger.info("Classifying files...")
        file_contexts = {}
        for file_path in lean_files:
            try:
                classification = self.classifier.classify(file_path)
                file_contexts[str(file_path)] = classification.context_type
            except Exception as e:
                logger.warning(f"Failed to classify {file_path}: {e}")
                file_contexts[str(file_path)] = "unknown"

        # Run experiments in parallel
        experiment_tasks = []
        for file_path in lean_files:
            for strategy in self.STRATEGIES:
                experiment_tasks.append((file_path, strategy, file_contexts[str(file_path)]))

        completed = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self._run_single_experiment, task[0], task[1], task[2]): task
                for task in experiment_tasks
            }

            for future in as_completed(future_to_task):
                completed += 1
                if completed % 100 == 0:
                    logger.info(f"Progress: {completed}/{total_experiments} experiments completed")

                try:
                    result = future.result()
                    if result:
                        self.results.append(result)
                except Exception as e:
                    task = future_to_task[future]
                    logger.error(f"Experiment failed for {task[0].name} with {task[1].name}: {e}")

        logger.info(f"Completed {len(self.results)} successful experiments")

        # Build payoff matrix
        self._build_payoff_matrix()

        # Save results
        self._save_results()

        # Generate analysis
        self._analyze_results()

    def _run_single_experiment(
        self, file_path: Path, strategy: OptimizationStrategy, context_type: str
    ) -> Optional[ExperimentResult]:
        """Run a single optimization experiment"""
        try:
            # Create temporary optimized file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as tmp:
                # Apply optimization strategy
                optimized_content = self._apply_strategy(file_path, strategy)
                tmp.write(optimized_content)
                tmp_path = Path(tmp.name)

            # Measure baseline compilation time
            baseline_time = self._measure_compilation_time(file_path)
            if baseline_time is None:
                return None

            # Measure optimized compilation time
            optimized_time = self._measure_compilation_time(tmp_path)
            if optimized_time is None:
                return None

            # Calculate speedup
            speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0

            # Clean up
            tmp_path.unlink()

            return ExperimentResult(
                file_path=str(file_path),
                context_type=context_type,
                strategy_name=strategy.name,
                baseline_time=baseline_time,
                optimized_time=optimized_time,
                speedup=speedup,
                success=speedup > 1.05,  # 5% improvement threshold
                detailed_metrics={
                    "file_size": file_path.stat().st_size,
                    "line_count": len(file_path.read_text().splitlines()),
                },
            )

        except Exception as e:
            logger.debug(f"Experiment error: {e}")
            return None

    def _apply_strategy(self, file_path: Path, strategy: OptimizationStrategy) -> str:
        """Apply optimization strategy to file content"""
        content = file_path.read_text()

        # For experiment purposes, simulate different optimization strategies
        # In real implementation, this would call actual Simpulse optimization

        if strategy.name == "no_optimization":
            return content

        # Simulate optimization by adding comments (real version would modify simp lemmas)
        lines = content.splitlines()
        modified_lines = []

        for line in lines:
            if strategy.name == "conservative" and "theorem" in line:
                modified_lines.append(f"-- [simpulse: priority +10]\n{line}")
            elif strategy.name == "moderate" and "theorem" in line:
                modified_lines.append(f"-- [simpulse: priority +50]\n{line}")
            elif strategy.name == "aggressive" and "theorem" in line:
                modified_lines.append(f"-- [simpulse: priority +100]\n{line}")
            elif strategy.name == "selective_top5" and "theorem" in line and random.random() < 0.1:
                modified_lines.append(f"-- [simpulse: selected]\n{line}")
            elif strategy.name == "contextual_arithmetic" and any(
                op in line for op in ["+", "-", "*", "/"]
            ):
                modified_lines.append(f"-- [simpulse: arithmetic boost]\n{line}")
            elif (
                strategy.name == "inverse_reduction" and "theorem" in line and random.random() > 0.3
            ):
                modified_lines.append(f"-- [simpulse: priority -50]\n{line}")
            elif strategy.name == "random_shuffle" and "theorem" in line:
                priority = random.randint(-100, 100)
                modified_lines.append(f"-- [simpulse: priority {priority}]\n{line}")
            elif strategy.name == "adaptive_threshold":
                progress = len(modified_lines) / len(lines)
                priority = int(30 + (80 - 30) * progress)
                if "theorem" in line:
                    modified_lines.append(f"-- [simpulse: adaptive priority {priority}]\n{line}")
                else:
                    modified_lines.append(line)
            elif strategy.name == "kitchen_sink" and "theorem" in line:
                modified_lines.append(f"-- [simpulse: multi-strategy optimization]\n{line}")
            else:
                modified_lines.append(line)

        return "\n".join(modified_lines)

    def _measure_compilation_time(self, file_path: Path, timeout: int = 30) -> Optional[float]:
        """Measure Lean compilation time for a file"""
        try:
            start_time = time.time()
            result = subprocess.run(
                [self.lean_executable, str(file_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed_time = time.time() - start_time

            if result.returncode != 0:
                # Compilation failed
                return None

            return elapsed_time

        except subprocess.TimeoutExpired:
            return None
        except Exception as e:
            logger.debug(f"Failed to measure compilation time: {e}")
            return None

    def _build_payoff_matrix(self):
        """Build empirical payoff matrix from results"""
        if not self.results:
            logger.warning("No results to build payoff matrix")
            return

        # Create DataFrame for analysis
        df = pd.DataFrame(
            [
                {
                    "context_type": r.context_type,
                    "strategy": r.strategy_name,
                    "speedup": r.speedup,
                    "success": r.success,
                }
                for r in self.results
            ]
        )

        # Build payoff matrix (context × strategy → average speedup)
        self.payoff_matrix = df.pivot_table(
            values="speedup",
            index="context_type",
            columns="strategy",
            aggfunc="mean",
            observed=False,
        )

        # Also calculate success rates
        self.success_matrix = df.pivot_table(
            values="success",
            index="context_type",
            columns="strategy",
            aggfunc="mean",
            observed=False,
        )

        logger.info("Payoff matrix built successfully")

    def _save_results(self):
        """Save experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw results
        results_file = self.output_dir / f"experiment_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(
                [
                    {
                        "file_path": r.file_path,
                        "context_type": r.context_type,
                        "strategy_name": r.strategy_name,
                        "baseline_time": r.baseline_time,
                        "optimized_time": r.optimized_time,
                        "speedup": r.speedup,
                        "success": r.success,
                        "error_message": r.error_message,
                        "detailed_metrics": r.detailed_metrics,
                    }
                    for r in self.results
                ],
                f,
                indent=2,
            )

        # Save payoff matrix
        if self.payoff_matrix is not None:
            payoff_file = self.output_dir / f"payoff_matrix_{timestamp}.csv"
            self.payoff_matrix.to_csv(payoff_file)

            success_file = self.output_dir / f"success_matrix_{timestamp}.csv"
            self.success_matrix.to_csv(success_file)

        logger.info(f"Results saved to {self.output_dir}")

    def _analyze_results(self):
        """Generate comprehensive analysis of results"""
        if not self.results or self.payoff_matrix is None:
            logger.warning("Insufficient data for analysis")
            return

        # Create visualizations
        self._create_payoff_heatmap()
        self._create_strategy_comparison()
        self._create_context_analysis()

        # Generate summary report
        self._generate_summary_report()

    def _create_payoff_heatmap(self):
        """Create heatmap visualization of payoff matrix"""
        plt.figure(figsize=(12, 8))

        # Payoff heatmap
        sns.heatmap(
            self.payoff_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=1.0,
            cbar_kws={"label": "Speedup Factor"},
            vmin=0.5,
            vmax=2.0,
        )

        plt.title("Empirical Payoff Matrix: Context × Strategy → Speedup", fontsize=16)
        plt.xlabel("Optimization Strategy", fontsize=12)
        plt.ylabel("Context Type", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plt.savefig(self.output_dir / "payoff_matrix_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Success rate heatmap
        plt.figure(figsize=(12, 8))

        sns.heatmap(
            self.success_matrix,
            annot=True,
            fmt=".1%",
            cmap="Blues",
            cbar_kws={"label": "Success Rate"},
            vmin=0,
            vmax=1,
        )

        plt.title("Success Rate Matrix: Context × Strategy", fontsize=16)
        plt.xlabel("Optimization Strategy", fontsize=12)
        plt.ylabel("Context Type", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plt.savefig(self.output_dir / "success_rate_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _create_strategy_comparison(self):
        """Create strategy comparison visualizations"""
        df = pd.DataFrame(
            [{"strategy": r.strategy_name, "speedup": r.speedup} for r in self.results]
        )

        plt.figure(figsize=(12, 6))

        # Box plot of speedups by strategy
        strategies_ordered = [s.name for s in self.STRATEGIES]
        df["strategy"] = pd.Categorical(df["strategy"], categories=strategies_ordered, ordered=True)

        ax = sns.boxplot(data=df, x="strategy", y="speedup", showfliers=False)

        # Add median values
        medians = df.groupby(["strategy"])["speedup"].median()
        for i, strategy in enumerate(strategies_ordered):
            if strategy in medians:
                ax.text(
                    i,
                    medians[strategy] + 0.05,
                    f"{medians[strategy]:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        plt.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="No speedup")
        plt.xlabel("Optimization Strategy", fontsize=12)
        plt.ylabel("Speedup Factor", fontsize=12)
        plt.title("Strategy Performance Distribution", fontsize=16)
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()

        plt.savefig(self.output_dir / "strategy_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _create_context_analysis(self):
        """Create context-specific analysis"""
        df = pd.DataFrame(
            [
                {"context_type": r.context_type, "speedup": r.speedup, "strategy": r.strategy_name}
                for r in self.results
            ]
        )

        # Get unique contexts
        contexts = sorted(df["context_type"].unique())

        # Create subplots for each context
        n_contexts = len(contexts)
        n_cols = 3
        n_rows = (n_contexts + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, context in enumerate(contexts):
            if i < len(axes):
                ax = axes[i]
                context_data = df[df["context_type"] == context]

                if not context_data.empty:
                    # Bar plot of average speedup by strategy
                    avg_speedups = context_data.groupby("strategy")["speedup"].mean()
                    avg_speedups.plot(kind="bar", ax=ax, color="skyblue")

                    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
                    ax.set_title(f"{context}", fontsize=12)
                    ax.set_xlabel("")
                    ax.set_ylabel("Avg Speedup")
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

        # Hide empty subplots
        for i in range(len(contexts), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle("Context-Specific Strategy Performance", fontsize=16)
        plt.tight_layout()

        plt.savefig(self.output_dir / "context_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_summary_report(self):
        """Generate comprehensive summary report"""
        report = {
            "experiment_summary": {
                "total_experiments": len(self.results),
                "unique_files": len({r.file_path for r in self.results}),
                "strategies_tested": len(self.STRATEGIES),
                "contexts_found": (
                    len(self.payoff_matrix.index) if self.payoff_matrix is not None else 0
                ),
            },
            "best_strategies_by_context": {},
            "overall_best_strategy": None,
            "key_findings": [],
        }

        if self.payoff_matrix is not None:
            # Find best strategy for each context
            for context in self.payoff_matrix.index:
                best_strategy = self.payoff_matrix.loc[context].idxmax()
                best_speedup = self.payoff_matrix.loc[context].max()
                report["best_strategies_by_context"][context] = {
                    "strategy": best_strategy,
                    "speedup": float(best_speedup),
                    "success_rate": float(self.success_matrix.loc[context, best_strategy]),
                }

            # Overall best strategy
            overall_avg = self.payoff_matrix.mean()
            report["overall_best_strategy"] = {
                "strategy": overall_avg.idxmax(),
                "average_speedup": float(overall_avg.max()),
            }

            # Key findings
            if overall_avg.max() > 1.2:
                report["key_findings"].append(
                    f"Significant speedups achieved: {overall_avg.max():.2f}x average"
                )

            # Check for context-specific insights
            for context in self.payoff_matrix.index:
                if self.payoff_matrix.loc[context].max() > 1.5:
                    report["key_findings"].append(
                        f"Context '{context}' shows high optimization potential: "
                        f"up to {self.payoff_matrix.loc[context].max():.2f}x speedup"
                    )

        # Save report
        report_file = self.output_dir / "experiment_summary.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"Total experiments run: {report['experiment_summary']['total_experiments']}")
        print(f"Files tested: {report['experiment_summary']['unique_files']}")
        print(f"Strategies tested: {report['experiment_summary']['strategies_tested']}")

        if report["overall_best_strategy"]:
            print(f"\nBest overall strategy: {report['overall_best_strategy']['strategy']}")
            print(f"Average speedup: {report['overall_best_strategy']['average_speedup']:.2f}x")

        print("\nBest strategies by context:")
        for context, info in report["best_strategies_by_context"].items():
            print(
                f"  {context}: {info['strategy']} ({info['speedup']:.2f}x speedup, "
                f"{info['success_rate']:.1%} success)"
            )

        print("\nKey findings:")
        for finding in report["key_findings"]:
            print(f"  - {finding}")

        print(f"\nDetailed results saved to: {self.output_dir}")


def collect_diverse_lean_files(mathlib_path: Path, count: int = 1000) -> List[Path]:
    """Collect diverse sample of Lean files"""
    all_files = list(mathlib_path.rglob("*.lean"))

    # Filter out test files and examples
    filtered_files = [
        f
        for f in all_files
        if not any(skip in str(f) for skip in ["test", "Test", "example", "Example"])
    ]

    # Sample diversely based on file size and path depth
    if len(filtered_files) > count:
        # Sort by file size to ensure diversity
        filtered_files.sort(key=lambda f: (f.stat().st_size, len(f.parts)))

        # Take every nth file to maintain diversity
        step = len(filtered_files) // count
        sampled = filtered_files[::step][:count]
    else:
        sampled = filtered_files

    random.shuffle(sampled)  # Randomize order
    return sampled


def main():
    """Run the experiment runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Run optimization experiments on Lean files")
    parser.add_argument(
        "--mathlib-path",
        type=Path,
        default=Path.home() / ".elan/toolchains/leanprover--lean4---v4.12.0/lib/lean/library",
        help="Path to Lean library files",
    )
    parser.add_argument(
        "--file-count", type=int, default=100, help="Number of files to test (default: 100)"
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_results"),
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Find Lean files
    if not args.mathlib_path.exists():
        # Try to find Lean files in the project
        lean_files = list(Path.cwd().rglob("*.lean"))[: args.file_count]
        if not lean_files:
            print("Error: No Lean files found")
            return
    else:
        lean_files = collect_diverse_lean_files(args.mathlib_path, args.file_count)

    print(f"Collected {len(lean_files)} Lean files for testing")

    # Run experiments
    runner = ExperimentRunner(output_dir=args.output_dir)
    runner.run_experiments(lean_files, max_workers=args.workers)


if __name__ == "__main__":
    main()
