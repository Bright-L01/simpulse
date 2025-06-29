"""Simpulse CLI - ML-powered simp rule optimization for Lean 4."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .analysis.health_checker import HealthChecker
from .optimization.optimizer import SimpOptimizer
from .profiling.benchmarker import Benchmarker

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="simpulse")
def cli():
    """Simpulse - Optimize Lean 4 simp rule priorities for better performance."""


@cli.command()
@click.argument(
    "project_path", type=click.Path(exists=True, path_type=Path), default="."
)
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
            console.print(
                f"\n💡 [yellow]High optimization potential detected![/yellow]"
            )
            console.print(
                f"   Run [cyan]simpulse optimize {project_path}[/cyan] to optimize\n"
            )
        else:
            console.print(f"\n✅ [green]Your project is well optimized![/green]\n")


@cli.command()
@click.argument(
    "project_path", type=click.Path(exists=True, path_type=Path), default="."
)
@click.option(
    "--output", "-o", type=click.Path(), help="Output path for optimization plan"
)
@click.option("--apply", is_flag=True, help="Apply optimizations directly")
@click.option(
    "--strategy",
    type=click.Choice(["conservative", "balanced", "aggressive"]),
    default="balanced",
    help="Optimization strategy",
)
def optimize(project_path: Path, output: Path, apply: bool, strategy: str):
    """Generate optimized simp rule priorities for your project."""
    console.print(f"\n🧠 Optimizing [cyan]{project_path}[/cyan]...\n")

    optimizer = SimpOptimizer(strategy=strategy)

    # Run optimization with progress bar
    with console.status("[bold green]Analyzing simp rules...") as status:
        analysis = optimizer.analyze(project_path)
        status.update("[bold green]Generating optimizations...")
        optimization = optimizer.optimize(analysis)

    console.print(f"\n✨ Optimization complete!")
    console.print(f"   Rules to optimize: [cyan]{optimization.rules_changed}[/cyan]")
    console.print(
        f"   Estimated improvement: [green]{optimization.estimated_improvement}%[/green]\n"
    )

    # Save or apply
    if apply:
        console.print("📝 Applying optimizations...")
        optimizer.apply(optimization, project_path)
        console.print("✅ Optimizations applied successfully!")
    elif output:
        optimization.save(output)
        console.print(f"💾 Optimization plan saved to: [cyan]{output}[/cyan]")
    else:
        # Show preview
        console.print("Preview (first 5 changes):")
        for i, change in enumerate(optimization.changes[:5]):
            console.print(
                f"  {change.rule_name}: {change.old_priority} → {change.new_priority}"
            )
        console.print("\nUse --apply to apply changes or -o to save plan")


@cli.command()
@click.argument(
    "project_path", type=click.Path(exists=True, path_type=Path), default="."
)
@click.option("--runs", "-r", type=int, default=3, help="Number of benchmark runs")
@click.option(
    "--compare", type=click.Path(exists=True), help="Compare with optimization plan"
)
def benchmark(project_path: Path, runs: int, compare: Path):
    """Run performance benchmarks on your Lean 4 project."""
    console.print(f"\n🏃 Benchmarking [cyan]{project_path}[/cyan]...\n")

    benchmarker = Benchmarker()

    if compare:
        # Run before/after comparison
        console.print("Running comparative benchmark...")
        results = benchmarker.compare(project_path, compare, runs=runs)

        improvement = results.improvement_percentage
        console.print(f"\n📊 Results:")
        console.print(f"   Baseline: {results.baseline_mean:.2f}s")
        console.print(f"   Optimized: {results.optimized_mean:.2f}s")
        console.print(
            f"   [{'green' if improvement > 0 else 'red'}]Improvement: {improvement:.1f}%[/]\n"
        )
    else:
        # Run single benchmark
        results = benchmarker.benchmark(project_path, runs=runs)
        console.print(f"\n📊 Benchmark Results:")
        console.print(f"   Mean build time: {results.mean:.2f}s")
        console.print(f"   Std deviation: {results.stdev:.2f}s\n")


@cli.command()
def version():
    """Show version information."""
    console.print("Simpulse v1.0.0")
    console.print("ML-powered simp rule optimization for Lean 4")
    console.print("https://github.com/Bright-L01/simpulse")


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
