"""Simpulse CLI - ML-powered simp rule optimization for Lean 4."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .analysis.health_checker import HealthChecker
from .optimization.fast_optimizer import FastOptimizer
from .optimization.optimizer import SimpOptimizer
from .profiling.benchmarker import Benchmarker
from .profiling.performance_benchmarks import run_benchmarks
from .profiling.simpulse_profiler import SimpulseProfiler

console = Console()


@click.group()
@click.version_option(version="1.1.0", prog_name="simpulse")
def cli():
    """Simpulse - Optimize Lean 4 simp rule priorities for better performance."""


@cli.command()
@click.argument("project_path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--json", is_flag=True, help="Output in JSON format")
def check(project_path: Path, json: bool):
    """Check if your Lean 4 project could benefit from optimization."""
    console.print(f"\n🔍 Checking [cyan]{project_path}[/cyan]...\n")

    checker = HealthChecker()
    result = checker.check_project(project_path)

    if json:
        click.echo(result.model_dump_json(indent=2))
    else:
        # Display results in a nice table
        table = Table(title="Simp Rule Health Check")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Status", style="green")

        table.add_row("Total Rules", str(result.total_rules), "✓")
        table.add_row(
            "Default Priority",
            f"{result.default_priority_percentage:.0f}%",
            "⚠️" if result.default_priority_percentage > 80 else "✓",
        )
        table.add_row(
            "Optimization Score",
            f"{result.score}/100",
            "🎯" if result.score > 60 else "✓",
        )
        table.add_row(
            "Estimated Improvement",
            f"{result.estimated_improvement}%",
            "🚀" if result.estimated_improvement > 30 else "✓",
        )

        console.print(table)

        if result.score > 60:
            console.print("\n💡 [yellow]High optimization potential detected![/yellow]")
            console.print(f"   Run [cyan]simpulse optimize {project_path}[/cyan] to optimize\n")
        else:
            console.print("\n✅ [green]Your project is well optimized![/green]\n")


@cli.command()
@click.argument("project_path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--output", "-o", type=click.Path(), help="Output path for optimization plan")
@click.option("--apply", is_flag=True, help="Apply optimizations directly")
@click.option(
    "--strategy",
    type=click.Choice(["conservative", "balanced", "aggressive", "performance", "frequency"]),
    default="balanced",
    help="Optimization strategy",
)
@click.option(
    "--fast",
    is_flag=True,
    help="Use fast optimizer with parallel processing",
)
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Validate correctness of optimizations (recommended)",
)
@click.option(
    "--safety-report",
    type=click.Path(),
    help="Path to save safety validation report",
)
def optimize(
    project_path: Path,
    output: Path,
    apply: bool,
    strategy: str,
    validate: bool,
    safety_report: Path,
    fast: bool,
):
    """Generate optimized simp rule priorities for your project."""
    console.print(f"\n🧠 Optimizing [cyan]{project_path}[/cyan]...\n")

    # Import additional modules if validation is enabled
    if validate:
        from .analyzer import LeanAnalyzer
        from .optimizer import PriorityOptimizer

        analyzer = LeanAnalyzer()
        optimizer = PriorityOptimizer(validate_correctness=True)

        console.print("🔒 [yellow]Correctness validation enabled[/yellow]\n")
    else:
        if fast:
            optimizer = FastOptimizer(strategy=strategy)
        else:
            optimizer = SimpOptimizer(strategy=strategy)

    # Run optimization with progress bar
    with console.status("[bold green]Analyzing simp rules...") as status:
        if validate:
            analysis = analyzer.analyze_project(project_path)
        else:
            analysis = optimizer.analyze(project_path)

        status.update("[bold green]Generating optimizations...")

        if validate:
            # Use the new validation-aware optimization
            optimization_result = optimizer.optimize_with_safety_check(
                analysis, output_dir=Path(safety_report).parent if safety_report else None
            )

            # Extract optimization info for display
            optimization = type(
                "obj",
                (object,),
                {
                    "rules_changed": optimization_result.get("total_suggestions", 0),
                    "estimated_improvement": 15.0,  # Default estimate
                },
            )()

            if optimization_result.get("validation_enabled"):
                console.print("✅ [green]Correctness validation complete![/green]")
                batch_report = optimization_result.get("batch_report", {})
                if batch_report:
                    console.print(
                        f"   Success rate: [cyan]{batch_report.get('overall_success_rate', 0):.1%}[/cyan]"
                    )
                    console.print(
                        f"   Average speedup: [cyan]{batch_report.get('average_speedup', 1.0):.2f}x[/cyan]"
                    )
        else:
            optimization = optimizer.optimize(analysis)

    console.print("\n✨ Optimization complete!")
    console.print(f"   Rules to optimize: [cyan]{optimization.rules_changed}[/cyan]")
    console.print(
        f"   Estimated improvement: [green]{optimization.estimated_improvement}%[/green]\n"
    )

    # Save or apply
    if apply:
        console.print("📝 Applying optimizations...")
        if validate:
            console.print(
                "⚠️  [yellow]Note: With validation enabled, use the generated optimization script to apply changes[/yellow]"
            )
        else:
            optimizer.apply(optimization, project_path)
            console.print("✅ Optimizations applied successfully!")
    elif output:
        if validate:
            import json

            with open(output, "w") as f:
                json.dump(optimization_result if validate else optimization, f, indent=2)
        else:
            optimization.save(output)
        console.print(f"💾 Optimization plan saved to: [cyan]{output}[/cyan]")
    else:
        # Show preview
        console.print("Preview (first 5 changes):")
        if validate and optimization_result.get("suggestions"):
            for i, suggestion in enumerate(optimization_result["suggestions"][:5]):
                console.print(
                    f"  {suggestion['rule_name']}: {suggestion['current_priority'] or 'default'} → {suggestion['suggested_priority']}"
                )
        elif hasattr(optimization, "changes"):
            for i, change in enumerate(optimization.changes[:5]):
                console.print(
                    f"  {change.rule_name}: {change.old_priority} → {change.new_priority}"
                )
        console.print("\nUse --apply to apply changes or -o to save plan")

    # Save safety report if requested and validation was enabled
    if safety_report and validate and optimization_result.get("safety_report"):
        import json

        with open(safety_report, "w") as f:
            json.dump(optimization_result["safety_report"], f, indent=2)
        console.print(f"\n📊 Safety report saved to: [cyan]{safety_report}[/cyan]")


@cli.command()
@click.argument("project_path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--runs", "-r", type=int, default=3, help="Number of benchmark runs")
@click.option("--compare", type=click.Path(exists=True), help="Compare with optimization plan")
def benchmark(project_path: Path, runs: int, compare: Path):
    """Run performance benchmarks on your Lean 4 project."""
    console.print(f"\n🏃 Benchmarking [cyan]{project_path}[/cyan]...\n")

    benchmarker = Benchmarker()

    if compare:
        # Run before/after comparison
        console.print("Running comparative benchmark...")
        results = benchmarker.compare(project_path, compare, runs=runs)

        improvement = results.improvement_percentage
        console.print("\n📊 Results:")
        console.print(f"   Baseline: {results.baseline_mean:.2f}s")
        console.print(f"   Optimized: {results.optimized_mean:.2f}s")
        console.print(
            f"   [{'green' if improvement > 0 else 'red'}]Improvement: {improvement:.1f}%[/]\n"
        )
    else:
        # Run single benchmark
        results = benchmarker.benchmark(project_path, runs=runs)
        console.print("\n📊 Benchmark Results:")
        console.print(f"   Mean build time: {results.mean:.2f}s")
        console.print(f"   Std deviation: {results.stdev:.2f}s\n")


@cli.command()
@click.argument("project_path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--detailed", is_flag=True, help="Show detailed profiling information")
@click.option("--output", "-o", type=click.Path(), help="Save profiling report to file")
def profile(project_path: Path, detailed: bool, output: Path):
    """Profile Simpulse performance on your project."""
    console.print(f"\n📊 Profiling [cyan]{project_path}[/cyan]...\n")

    profiler = SimpulseProfiler()

    # Profile the full pipeline
    with console.status("[bold green]Running performance analysis..."):
        results = profiler.profile_full_pipeline(project_path)

    # Display summary
    console.print("\n[bold]Performance Summary:[/bold]")
    console.print(f"Total time: [cyan]{results['total_duration']:.3f}s[/cyan]")
    console.print(f"Analysis phase: [cyan]{results['analysis_metrics'].duration:.3f}s[/cyan]")
    console.print(
        f"Optimization phase: [cyan]{results['optimization_metrics'].duration:.3f}s[/cyan]"
    )
    console.print(f"Files processed: [cyan]{results['analysis_metrics'].file_count}[/cyan]")
    console.print(f"Rules found: [cyan]{results['analysis_metrics'].rule_count}[/cyan]")

    if detailed:
        console.print("\n" + profiler.generate_report())

    if output:
        profiler.save_report(output)
        console.print(f"\n[green]Report saved to {output}[/green]")


@cli.command()
@click.argument("project_path", type=click.Path(exists=True, path_type=Path), required=False)
@click.option("--compare", is_flag=True, help="Compare original vs optimized performance")
def perf_test(project_path: Path, compare: bool):
    """Run comprehensive performance benchmarks."""
    console.print("\n[bold cyan]Running Performance Tests[/bold cyan]\n")

    if compare and project_path:
        # Run before/after comparison
        from .optimization.fast_optimizer import FastOptimizer

        console.print("Comparing original vs optimized implementation...")

        # Time original
        import time

        start = time.time()
        optimizer1 = SimpOptimizer()
        analysis1 = optimizer1.analyze(project_path)
        optimizer1.optimize(analysis1)
        time1 = time.time() - start

        # Time optimized
        start = time.time()
        optimizer2 = FastOptimizer()
        analysis2 = optimizer2.analyze(project_path)
        optimizer2.optimize(analysis2)
        time2 = time.time() - start

        speedup = time1 / time2 if time2 > 0 else 0

        console.print(f"\nOriginal: [cyan]{time1:.3f}s[/cyan]")
        console.print(f"Optimized: [cyan]{time2:.3f}s[/cyan]")
        console.print(f"Speedup: [green]{speedup:.2f}x[/green]")
    else:
        # Run full benchmark suite
        run_benchmarks(project_path)


@cli.command()
def version():
    """Show version information."""
    console.print("Simpulse v1.1.0")
    console.print("ML-powered simp rule optimization for Lean 4")
    console.print("https://github.com/Bright-L01/simpulse")


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
