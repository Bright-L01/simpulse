#!/usr/bin/env python3
"""
Deep Statistical Analysis of Empirical Optimization Results

Extracts actionable insights from experimental data to build
a statistical optimization model based on facts, not theory.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")


@dataclass
class StrategyProfile:
    """Statistical profile of an optimization strategy"""

    name: str
    mean_speedup: float
    median_speedup: float
    std_deviation: float
    variance: float
    success_rate: float
    risk_score: float  # 0-1, higher = riskier
    consistency_score: float  # 0-1, higher = more consistent
    percentile_5: float
    percentile_95: float
    iqr: float  # Interquartile range
    cv: float  # Coefficient of variation
    sharpe_ratio: float  # Risk-adjusted return

    @property
    def is_safe(self) -> bool:
        """Safe = consistent gains with low variance"""
        return self.consistency_score > 0.7 and self.risk_score < 0.3

    @property
    def is_risky(self) -> bool:
        """Risky = high variance, unpredictable outcomes"""
        return self.risk_score > 0.7 or self.cv > 0.5


@dataclass
class ContextProfile:
    """Statistical profile of a context type"""

    name: str
    optimization_resistance: float  # 0-1, higher = more resistant
    best_strategy: str
    best_speedup: float
    worst_strategy: str
    worst_speedup: float
    strategy_rankings: Dict[str, int]
    variance_across_strategies: float
    predictability_score: float  # 0-1, higher = more predictable


@dataclass
class StatisticalOptimizationModel:
    """Statistical model built from empirical data"""

    strategy_profiles: Dict[str, StrategyProfile]
    context_profiles: Dict[str, ContextProfile]
    payoff_matrix: pd.DataFrame
    variance_matrix: pd.DataFrame
    confidence_intervals: Dict[Tuple[str, str], Tuple[float, float]]

    def recommend_strategy(self, context: str, risk_tolerance: float = 0.5) -> str:
        """Recommend strategy based on context and risk tolerance"""
        if context not in self.context_profiles:
            return "no_optimization"

        # Get candidate strategies
        candidates = []
        for strategy, profile in self.strategy_profiles.items():
            expected_speedup = self.payoff_matrix.loc[context, strategy]

            # Risk-adjusted score
            if risk_tolerance < 0.3:  # Conservative
                if profile.is_safe:
                    score = expected_speedup * profile.consistency_score
                    candidates.append((strategy, score))
            elif risk_tolerance > 0.7:  # Aggressive
                score = expected_speedup * (1 + profile.risk_score)
                candidates.append((strategy, score))
            else:  # Balanced
                score = expected_speedup * profile.sharpe_ratio
                candidates.append((strategy, score))

        # Return best strategy
        if candidates:
            return max(candidates, key=lambda x: x[1])[0]
        return self.context_profiles[context].best_strategy


class EmpiricalAnalyzer:
    """Deep analysis of empirical optimization results"""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.raw_data = []
        self.df = None
        self.model = None

    def load_results(self) -> bool:
        """Load experimental results from directory"""
        # Find the most recent results file
        result_files = list(self.results_dir.glob("experiment_results_*.json"))
        if not result_files:
            print(f"No result files found in {self.results_dir}")
            return False

        latest_file = max(result_files, key=lambda f: f.stat().st_mtime)

        with open(latest_file) as f:
            self.raw_data = json.load(f)

        # Convert to DataFrame
        self.df = pd.DataFrame(self.raw_data)

        print(f"Loaded {len(self.raw_data)} experimental results")
        print(f"Contexts: {self.df['context_type'].nunique()}")
        print(f"Strategies: {self.df['strategy_name'].nunique()}")

        return True

    def analyze(self) -> StatisticalOptimizationModel:
        """Perform comprehensive statistical analysis"""
        print("\n🔬 DEEP STATISTICAL ANALYSIS")
        print("=" * 60)

        # 1. Analyze strategies
        print("\n1. STRATEGY ANALYSIS")
        print("-" * 40)
        strategy_profiles = self._analyze_strategies()

        # 2. Analyze contexts
        print("\n2. CONTEXT ANALYSIS")
        print("-" * 40)
        context_profiles = self._analyze_contexts()

        # 3. Build statistical matrices
        print("\n3. STATISTICAL MATRICES")
        print("-" * 40)
        payoff_matrix, variance_matrix = self._build_matrices()

        # 4. Calculate confidence intervals
        print("\n4. CONFIDENCE INTERVALS")
        print("-" * 40)
        confidence_intervals = self._calculate_confidence_intervals()

        # Create model first
        self.model = StatisticalOptimizationModel(
            strategy_profiles=strategy_profiles,
            context_profiles=context_profiles,
            payoff_matrix=payoff_matrix,
            variance_matrix=variance_matrix,
            confidence_intervals=confidence_intervals,
        )

        # 5. Identify patterns
        print("\n5. PATTERN IDENTIFICATION")
        print("-" * 40)
        self._identify_patterns()

        # 6. Generate insights
        print("\n6. KEY INSIGHTS")
        print("-" * 40)
        self._generate_insights()

        return self.model

    def _analyze_strategies(self) -> Dict[str, StrategyProfile]:
        """Deep analysis of each strategy"""
        profiles = {}

        for strategy in self.df["strategy_name"].unique():
            strategy_data = self.df[self.df["strategy_name"] == strategy]["speedup"]

            # Calculate statistics
            mean_speedup = strategy_data.mean()
            median_speedup = strategy_data.median()
            std_dev = strategy_data.std()
            variance = strategy_data.var()

            # Success metrics
            success_rate = (strategy_data > 1.05).mean()  # 5% improvement threshold

            # Risk metrics
            percentile_5 = strategy_data.quantile(0.05)
            percentile_95 = strategy_data.quantile(0.95)
            iqr = strategy_data.quantile(0.75) - strategy_data.quantile(0.25)
            cv = std_dev / mean_speedup if mean_speedup > 0 else np.inf

            # Downside risk
            downside_data = strategy_data[strategy_data < 1.0]
            downside_risk = len(downside_data) / len(strategy_data)

            # Risk score (0-1) - handle NaN CV
            if np.isnan(cv) or np.isinf(cv):
                risk_score = downside_risk
                cv = 0.0
            else:
                risk_score = min(1.0, cv + downside_risk)

            # Consistency score (0-1)
            if np.isnan(cv) or cv == 0:
                consistency_score = 1.0 - downside_risk
            else:
                consistency_score = 1 / (1 + cv) * (1 - downside_risk)

            # Sharpe ratio (risk-adjusted return)
            excess_return = mean_speedup - 1.0  # baseline = 1.0
            sharpe_ratio = excess_return / std_dev if std_dev > 0 else 0

            profile = StrategyProfile(
                name=strategy,
                mean_speedup=mean_speedup,
                median_speedup=median_speedup,
                std_deviation=std_dev,
                variance=variance,
                success_rate=success_rate,
                risk_score=risk_score,
                consistency_score=consistency_score,
                percentile_5=percentile_5,
                percentile_95=percentile_95,
                iqr=iqr,
                cv=cv,
                sharpe_ratio=sharpe_ratio,
            )

            profiles[strategy] = profile

            # Print summary
            print(f"\n{strategy}:")
            print(f"  Mean speedup: {mean_speedup:.3f}x")
            print(f"  Success rate: {success_rate:.1%}")
            print(
                f"  Risk score: {risk_score:.2f} {'⚠️ RISKY' if profile.is_risky else '✅ SAFE' if profile.is_safe else ''}"
            )
            print(f"  Consistency: {consistency_score:.2f}")
            print(f"  95% CI: [{percentile_5:.2f}, {percentile_95:.2f}]")

        return profiles

    def _analyze_contexts(self) -> Dict[str, ContextProfile]:
        """Deep analysis of each context type"""
        profiles = {}

        for context in self.df["context_type"].unique():
            context_data = self.df[self.df["context_type"] == context]

            # Strategy performance in this context
            strategy_speedups = context_data.groupby("strategy_name")["speedup"].agg(
                ["mean", "std"]
            )

            # Best and worst strategies
            best_strategy = strategy_speedups["mean"].idxmax()
            best_speedup = strategy_speedups["mean"].max()
            worst_strategy = strategy_speedups["mean"].idxmin()
            worst_speedup = strategy_speedups["mean"].min()

            # Strategy rankings
            rankings = strategy_speedups["mean"].sort_values(ascending=False)
            strategy_rankings = {s: i + 1 for i, s in enumerate(rankings.index)}

            # Variance across strategies
            variance_across = strategy_speedups["mean"].var()

            # Optimization resistance (lower speedups = more resistant)
            avg_improvement = strategy_speedups["mean"].mean() - 1.0
            optimization_resistance = 1 / (1 + max(0, avg_improvement))

            # Predictability (lower variance = more predictable)
            avg_variance = strategy_speedups["std"].mean()
            if np.isnan(avg_variance):
                predictability_score = 1.0
            else:
                predictability_score = 1 / (1 + avg_variance)

            profile = ContextProfile(
                name=context,
                optimization_resistance=optimization_resistance,
                best_strategy=best_strategy,
                best_speedup=best_speedup,
                worst_strategy=worst_strategy,
                worst_speedup=worst_speedup,
                strategy_rankings=strategy_rankings,
                variance_across_strategies=variance_across,
                predictability_score=predictability_score,
            )

            profiles[context] = profile

            # Print summary
            print(f"\n{context}:")
            print(f"  Best: {best_strategy} ({best_speedup:.2f}x)")
            print(f"  Worst: {worst_strategy} ({worst_speedup:.2f}x)")
            print(
                f"  Resistance: {optimization_resistance:.2f} {'🛡️ RESISTANT' if optimization_resistance > 0.8 else ''}"
            )
            print(f"  Predictability: {predictability_score:.2f}")

        return profiles

    def _build_matrices(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build payoff and variance matrices"""
        # Payoff matrix (mean speedups)
        payoff_matrix = self.df.pivot_table(
            values="speedup", index="context_type", columns="strategy_name", aggfunc="mean"
        )

        # Variance matrix
        variance_matrix = self.df.pivot_table(
            values="speedup", index="context_type", columns="strategy_name", aggfunc="var"
        )

        print(f"Payoff matrix shape: {payoff_matrix.shape}")
        print(f"Average speedup across all: {payoff_matrix.mean().mean():.3f}x")

        return payoff_matrix, variance_matrix

    def _calculate_confidence_intervals(self) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """Calculate 95% confidence intervals for each context-strategy pair"""
        intervals = {}

        for context in self.df["context_type"].unique():
            for strategy in self.df["strategy_name"].unique():
                data = self.df[
                    (self.df["context_type"] == context) & (self.df["strategy_name"] == strategy)
                ]["speedup"]

                if len(data) > 1:
                    # Calculate 95% CI
                    mean = data.mean()
                    sem = stats.sem(data)
                    ci = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=sem)
                    intervals[(context, strategy)] = ci
                elif len(data) == 1:
                    # Single data point
                    value = data.iloc[0]
                    intervals[(context, strategy)] = (value, value)

        return intervals

    def _identify_patterns(self):
        """Identify key patterns in the data"""
        print("\n🔍 Pattern Discovery:")

        # 1. Consistent winners
        if self.model:
            payoff_matrix = self.model.payoff_matrix

            # Strategies that win in most contexts
            win_counts = {}
            for strategy in payoff_matrix.columns:
                wins = sum(payoff_matrix[strategy] == payoff_matrix.max(axis=1))
                win_counts[strategy] = wins

            print("\n📊 Strategy Win Counts:")
            for strategy, wins in sorted(win_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {strategy}: {wins}/{len(payoff_matrix)} contexts")

        # 2. High variance strategies
        print("\n⚡ High Variance Strategies:")
        for name, profile in sorted(
            self.model.strategy_profiles.items(), key=lambda x: x[1].cv, reverse=True
        )[:3]:
            print(f"  {name}: CV={profile.cv:.2f}, σ={profile.std_deviation:.3f}")

        # 3. Optimization resistant contexts
        print("\n🛡️ Optimization Resistant Contexts:")
        for name, profile in sorted(
            self.model.context_profiles.items(),
            key=lambda x: x[1].optimization_resistance,
            reverse=True,
        )[:3]:
            print(f"  {name}: resistance={profile.optimization_resistance:.2f}")

    def _generate_insights(self):
        """Generate actionable insights from analysis"""
        insights = []

        # Safe strategies
        safe_strategies = [
            (name, profile)
            for name, profile in self.model.strategy_profiles.items()
            if profile.is_safe
        ]
        if safe_strategies:
            insights.append(
                f"Safe strategies with consistent gains: "
                f"{', '.join(s[0] for s in safe_strategies)}"
            )

        # Risky strategies
        risky_strategies = [
            (name, profile)
            for name, profile in self.model.strategy_profiles.items()
            if profile.is_risky
        ]
        if risky_strategies:
            insights.append(
                f"Risky strategies with high variance: "
                f"{', '.join(s[0] for s in risky_strategies)}"
            )

        # Best overall
        best_overall = max(self.model.strategy_profiles.items(), key=lambda x: x[1].sharpe_ratio)
        insights.append(
            f"Best risk-adjusted strategy: {best_overall[0]} "
            f"(Sharpe ratio: {best_overall[1].sharpe_ratio:.2f})"
        )

        # Resistant contexts
        resistant = [
            name
            for name, profile in self.model.context_profiles.items()
            if profile.optimization_resistance > 0.8
        ]
        if resistant:
            insights.append(f"Optimization-resistant contexts: {', '.join(resistant)}")

        print("\n💡 Key Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"  {i}. {insight}")

    def visualize_results(self):
        """Create comprehensive visualizations"""
        if not self.model:
            print("No model available. Run analyze() first.")
            return

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))

        # 1. Strategy risk-return scatter
        ax1 = plt.subplot(2, 3, 1)
        self._plot_risk_return(ax1)

        # 2. Variance heatmap
        ax2 = plt.subplot(2, 3, 2)
        self._plot_variance_heatmap(ax2)

        # 3. Strategy consistency
        ax3 = plt.subplot(2, 3, 3)
        self._plot_consistency_chart(ax3)

        # 4. Context resistance
        ax4 = plt.subplot(2, 3, 4)
        self._plot_resistance_chart(ax4)

        # 5. Win rate matrix
        ax5 = plt.subplot(2, 3, 5)
        self._plot_win_matrix(ax5)

        # 6. Confidence intervals
        ax6 = plt.subplot(2, 3, 6)
        self._plot_confidence_intervals(ax6)

        plt.tight_layout()
        output_path = self.results_dir / "deep_analysis_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n📊 Visualizations saved to: {output_path}")
        plt.close()

    def _plot_risk_return(self, ax):
        """Risk-return scatter plot"""
        strategies = []
        returns = []
        risks = []
        colors = []

        for name, profile in self.model.strategy_profiles.items():
            strategies.append(name)
            returns.append(profile.mean_speedup - 1.0)  # Excess return
            risks.append(profile.std_deviation)

            if profile.is_safe:
                colors.append("green")
            elif profile.is_risky:
                colors.append("red")
            else:
                colors.append("orange")

        scatter = ax.scatter(risks, returns, c=colors, s=100, alpha=0.7)

        # Add labels
        for i, name in enumerate(strategies):
            ax.annotate(name, (risks[i], returns[i]), fontsize=8, alpha=0.7)

        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax.set_xlabel("Risk (Std Dev)")
        ax.set_ylabel("Excess Return")
        ax.set_title("Strategy Risk-Return Profile")
        ax.grid(True, alpha=0.3)

    def _plot_variance_heatmap(self, ax):
        """Variance heatmap"""
        sns.heatmap(
            self.model.variance_matrix,
            annot=True,
            fmt=".3f",
            cmap="Reds",
            ax=ax,
            cbar_kws={"label": "Variance"},
        )
        ax.set_title("Strategy Variance by Context")
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Context")

    def _plot_consistency_chart(self, ax):
        """Strategy consistency scores"""
        strategies = []
        consistency_scores = []
        colors = []

        for name, profile in sorted(
            self.model.strategy_profiles.items(), key=lambda x: x[1].consistency_score, reverse=True
        ):
            strategies.append(name)
            consistency_scores.append(profile.consistency_score)
            colors.append(
                "green" if profile.is_safe else "orange" if not profile.is_risky else "red"
            )

        bars = ax.bar(range(len(strategies)), consistency_scores, color=colors, alpha=0.7)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45, ha="right")
        ax.set_ylabel("Consistency Score")
        ax.set_title("Strategy Consistency Ranking")
        ax.axhline(y=0.7, color="green", linestyle="--", alpha=0.5, label="Safe threshold")
        ax.legend()

    def _plot_resistance_chart(self, ax):
        """Context optimization resistance"""
        contexts = []
        resistance_scores = []

        for name, profile in sorted(
            self.model.context_profiles.items(),
            key=lambda x: x[1].optimization_resistance,
            reverse=True,
        ):
            contexts.append(name)
            resistance_scores.append(profile.optimization_resistance)

        colors = ["red" if r > 0.8 else "orange" if r > 0.6 else "green" for r in resistance_scores]

        ax.barh(range(len(contexts)), resistance_scores, color=colors, alpha=0.7)
        ax.set_yticks(range(len(contexts)))
        ax.set_yticklabels(contexts)
        ax.set_xlabel("Optimization Resistance")
        ax.set_title("Context Optimization Resistance")
        ax.axvline(x=0.8, color="red", linestyle="--", alpha=0.5)

    def _plot_win_matrix(self, ax):
        """Strategy win counts by context"""
        # Calculate win matrix
        win_matrix = pd.DataFrame(
            0, index=self.model.payoff_matrix.index, columns=self.model.payoff_matrix.columns
        )

        for context in self.model.payoff_matrix.index:
            best_strategy = self.model.payoff_matrix.loc[context].idxmax()
            win_matrix.loc[context, best_strategy] = 1

        sns.heatmap(
            win_matrix, annot=True, fmt="d", cmap="Blues", ax=ax, cbar_kws={"label": "Wins"}
        )
        ax.set_title("Strategy Wins by Context")
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Context")

    def _plot_confidence_intervals(self, ax):
        """Confidence interval comparison"""
        # Select a representative context
        contexts = list(self.model.context_profiles.keys())
        if contexts:
            context = contexts[0]

            strategies = []
            means = []
            lower_bounds = []
            upper_bounds = []

            for strategy in self.model.strategy_profiles.keys():
                if (context, strategy) in self.model.confidence_intervals:
                    ci = self.model.confidence_intervals[(context, strategy)]
                    mean = self.model.payoff_matrix.loc[context, strategy]

                    strategies.append(strategy)
                    means.append(mean)
                    lower_bounds.append(mean - ci[0])
                    upper_bounds.append(ci[1] - mean)

            x = np.arange(len(strategies))
            ax.errorbar(x, means, yerr=[lower_bounds, upper_bounds], fmt="o", capsize=5, capthick=2)

            ax.set_xticks(x)
            ax.set_xticklabels(strategies, rotation=45, ha="right")
            ax.set_ylabel("Speedup")
            ax.set_title(f"95% Confidence Intervals ({context})")
            ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.3)

    def export_model(self, output_file: Path = None):
        """Export statistical model for production use"""
        if not self.model:
            print("No model to export. Run analyze() first.")
            return

        if output_file is None:
            output_file = self.results_dir / "statistical_optimization_model.json"

        # Convert model to JSON-serializable format
        model_data = {
            "strategy_profiles": {
                name: {
                    "mean_speedup": profile.mean_speedup,
                    "std_deviation": profile.std_deviation,
                    "success_rate": profile.success_rate,
                    "risk_score": profile.risk_score,
                    "consistency_score": profile.consistency_score,
                    "is_safe": profile.is_safe,
                    "is_risky": profile.is_risky,
                    "sharpe_ratio": profile.sharpe_ratio,
                }
                for name, profile in self.model.strategy_profiles.items()
            },
            "context_profiles": {
                name: {
                    "optimization_resistance": profile.optimization_resistance,
                    "best_strategy": profile.best_strategy,
                    "best_speedup": profile.best_speedup,
                    "strategy_rankings": profile.strategy_rankings,
                    "predictability_score": profile.predictability_score,
                }
                for name, profile in self.model.context_profiles.items()
            },
            "payoff_matrix": self.model.payoff_matrix.to_dict(),
            "variance_matrix": self.model.variance_matrix.to_dict(),
        }

        with open(output_file, "w") as f:
            json.dump(model_data, f, indent=2)

        print(f"\n📦 Model exported to: {output_file}")

        # Also export strategy recommendations
        recommendations = []
        for context in self.model.context_profiles.keys():
            rec = {
                "context": context,
                "conservative": self.model.recommend_strategy(context, risk_tolerance=0.2),
                "balanced": self.model.recommend_strategy(context, risk_tolerance=0.5),
                "aggressive": self.model.recommend_strategy(context, risk_tolerance=0.8),
            }
            recommendations.append(rec)

        rec_file = self.results_dir / "strategy_recommendations.json"
        with open(rec_file, "w") as f:
            json.dump(recommendations, f, indent=2)

        print(f"📋 Recommendations exported to: {rec_file}")


def main():
    """Run deep analysis on experimental results"""
    import argparse

    parser = argparse.ArgumentParser(description="Deep analysis of empirical optimization results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("simple_experiment"),
        help="Directory containing experimental results",
    )
    parser.add_argument(
        "--export", action="store_true", help="Export statistical model for production use"
    )

    args = parser.parse_args()

    print("🔬 EMPIRICAL OPTIMIZATION ANALYZER")
    print("=" * 60)

    analyzer = EmpiricalAnalyzer(args.results_dir)

    if not analyzer.load_results():
        print("Failed to load results. Run experiments first.")
        return

    # Perform deep analysis
    analyzer.analyze()

    # Create visualizations
    analyzer.visualize_results()

    # Export if requested
    if args.export:
        analyzer.export_model()

    print("\n✅ Analysis complete!")
    print(f"📊 Results saved to: {args.results_dir}/")


if __name__ == "__main__":
    main()
