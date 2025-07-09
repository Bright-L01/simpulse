"""Simple, direct CLI for Simpulse - no frameworks, just functionality."""

import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .error import handle_error
from .performance_guarantee import PerformanceGuarantee
from .unified_optimizer import UnifiedOptimizer

console = Console()

# Global flags
DEBUG_MODE = False
QUIET_MODE = False
VERBOSE_MODE = False


def print_success(message: str, details: str = None):
    """Print success message with consistent formatting."""
    if QUIET_MODE:
        return
    console.print(f"✅ {message}", style="green")
    if details and VERBOSE_MODE:
        console.print(f"   {details}", style="dim green")


def print_warning(message: str, suggestion: str = None):
    """Print warning message with optional suggestion."""
    if QUIET_MODE:
        return
    console.print(f"⚠️  {message}", style="yellow")
    if suggestion:
        console.print(f"💡 {suggestion}", style="dim yellow")


def print_error(message: str, suggestion: str = None):
    """Print error message with optional suggestion."""
    if QUIET_MODE:
        return
    console.print(f"❌ {message}", style="red")
    if suggestion:
        console.print(f"💡 {suggestion}", style="dim cyan")


def print_info(message: str, details: str = None):
    """Print info message with optional details."""
    if QUIET_MODE:
        return
    console.print(f"ℹ️  {message}", style="cyan")
    if details and VERBOSE_MODE:
        console.print(f"   {details}", style="dim cyan")


def get_error_suggestion(error: Exception, context: str = None) -> str:
    """Get helpful suggestion based on error type."""
    error_str = str(error).lower()

    if "file too large" in error_str:
        return "Try splitting large files or increase SIMPULSE_MAX_FILE_SIZE"
    elif "permission denied" in error_str:
        return "Check file permissions or run with appropriate privileges"
    elif "no such file" in error_str:
        return "Verify the path exists and contains Lean files"
    elif "timeout" in error_str:
        return "Try optimizing smaller directories or increase SIMPULSE_TIMEOUT"
    elif "memory" in error_str:
        return "Close other applications or increase SIMPULSE_MAX_MEMORY"
    elif "no lean files" in error_str:
        return "Ensure you're in a Lean project directory with .lean files"
    elif context == "optimization" and "no rules" in error_str:
        return "This project might already be optimized or have no @[simp] rules"
    else:
        return "Use --debug for more detailed error information"


def create_progress_bar():
    """Create a beautiful progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


@click.group()
@click.version_option(version="0.1.0-experimental", prog_name="simpulse")
@click.option("--debug", is_flag=True, help="Enable debug mode with detailed error messages")
@click.option("--quiet", "-q", is_flag=True, help="Quiet mode - minimal output")
@click.option("--verbose", "-v", is_flag=True, help="Verbose mode - detailed progress")
def cli(debug, quiet, verbose):
    """✨ Simpulse - Experimental optimizer for Lean 4 simp rules

    Adjusts simp rule priorities based on usage frequency (unverified performance claims).
    """
    global DEBUG_MODE, QUIET_MODE, VERBOSE_MODE
    DEBUG_MODE = debug
    QUIET_MODE = quiet
    VERBOSE_MODE = verbose

    # Quiet overrides verbose
    if quiet:
        VERBOSE_MODE = False
        console.quiet = True

    # Setup logging
    if quiet:
        log_level = logging.ERROR
    elif debug or verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format=(
            "%(levelname)s: %(message)s" if not debug else "%(levelname)s [%(name)s]: %(message)s"
        ),
        stream=sys.stderr,
    )


@cli.command()
@click.argument("project_path", type=click.Path(exists=True, path_type=Path), default=".")
def check(project_path: Path):
    """Check if your project could benefit from optimization."""
    if not QUIET_MODE:
        console.print()
        console.print(
            Panel(
                f"🔍 Analyzing [cyan]{project_path}[/cyan]",
                title="Simpulse Check",
                border_style="cyan",
            )
        )
        console.print()

    try:
        # Quick analysis with progress
        with create_progress_bar() as progress:
            task = progress.add_task("Scanning for simp rules...", total=100)

            optimizer = UnifiedOptimizer()
            progress.update(task, completed=30, description="Finding Lean files...")

            results = optimizer.optimize(project_path, apply=False)
            progress.update(task, completed=100, description="Analysis complete!")

        total_rules = results["total_rules"]
        optimizable = results["rules_changed"]
        improvement = results["estimated_improvement"]

        if total_rules == 0:
            print_warning(
                "No simp rules found", "Ensure you're in a Lean project with @[simp] annotations"
            )
            sys.exit(1)
        elif optimizable > 0:
            print_success(f"Found {total_rules} simp rules")
            print_info(
                f"Can optimize {optimizable} rules", f"Potential speedup: {improvement:.1f}%"
            )

            if not QUIET_MODE:
                console.print(
                    f"\n💫 Run [bold green]simpulse optimize[/bold green] to apply optimizations"
                )
        else:
            print_success(f"Found {total_rules} simp rules")
            print_info("Rules are already well-optimized!", "No performance improvements available")

    except Exception as e:
        error_msg = handle_error(e, debug=DEBUG_MODE)
        suggestion = get_error_suggestion(e, "check")
        print_error(f"Analysis failed: {error_msg}", suggestion)
        sys.exit(1)


@cli.command()
@click.argument("project_path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option(
    "-s",
    "--strategy",
    type=click.Choice(["frequency", "balanced", "conservative"]),
    default="frequency",
    help="Optimization strategy",
)
@click.option(
    "-o", "--output", type=click.Path(path_type=Path), help="Save optimization plan to file"
)
@click.option("--apply", is_flag=True, help="Apply optimizations immediately")
@click.option("--json", "as_json", is_flag=True, help="Output in JSON format")
def optimize(project_path: Path, strategy: str, output: Path, apply: bool, as_json: bool):
    """🚀 Optimize simp rule priorities for faster proof search."""

    # JSON mode bypasses all beautiful UI
    if as_json:
        try:
            optimizer = UnifiedOptimizer(strategy=strategy)
            results = optimizer.optimize(project_path, apply=apply)
            click.echo(json.dumps(results, indent=2))
        except Exception as e:
            error_result = {"error": str(e), "success": False}
            click.echo(json.dumps(error_result, indent=2))
            sys.exit(1)
        return

    # Beautiful UI mode
    if not QUIET_MODE:
        console.print()
        panel_title = "🚀 Optimization" if not apply else "⚡ Apply Optimization"
        console.print(
            Panel(
                f"Project: [cyan]{project_path}[/cyan]\n"
                f"Strategy: [yellow]{strategy}[/yellow]\n"
                f"Mode: [green]{'Apply changes' if apply else 'Preview only'}[/green]",
                title=panel_title,
                border_style="green" if apply else "yellow",
            )
        )
        console.print()

    try:
        # Multi-stage optimization with progress
        with create_progress_bar() as progress:
            # Stage 1: Initialize
            init_task = progress.add_task("Initializing optimizer...", total=100)
            optimizer = UnifiedOptimizer(strategy=strategy)
            progress.update(init_task, completed=100)

            # Stage 2: Scan files
            scan_task = progress.add_task("Scanning Lean files...", total=100)
            progress.update(scan_task, completed=50)

            # Stage 3: Analyze rules
            analyze_task = progress.add_task("Analyzing simp rules...", total=100)

            # Stage 4: Optimize
            optimize_task = progress.add_task("Computing optimizations...", total=100)

            results = optimizer.optimize(project_path, apply=apply)

            # Complete all tasks
            progress.update(scan_task, completed=100)
            progress.update(analyze_task, completed=100)
            progress.update(optimize_task, completed=100)

        # Results display
        total_rules = results["total_rules"]
        optimized = results["rules_changed"]
        improvement = results["estimated_improvement"]

        if total_rules == 0:
            print_warning("No simp rules found", "Ensure your project has @[simp] annotations")
            sys.exit(1)

        # Success message with theoretical estimate
        if optimized > 0:
            estimate_msg = f"{improvement:.1f}% theoretical improvement (unverified)"
            print_success(f"Optimization complete! {estimate_msg}")
            print_info(f"Optimized {optimized} of {total_rules} rules")

            if VERBOSE_MODE:
                print_info(f"Strategy: {strategy}", f"Applied: {'Yes' if apply else 'No'}")

            # Beautiful results table
            if results["changes"] and not QUIET_MODE:
                console.print()
                table = Table(title=f"✨ Optimization Results", title_style="bold green")
                table.add_column("Rule", style="cyan", no_wrap=True)
                table.add_column("Before", style="red", justify="center")
                table.add_column("After", style="green", justify="center")
                table.add_column("Impact", style="yellow")

                for change in results["changes"][:10]:
                    old_priority = change["old_priority"]
                    new_priority = change["new_priority"]

                    # Calculate improvement direction
                    if new_priority < old_priority:
                        impact = "🚀 Faster"
                    else:
                        impact = "⚡ Optimized"

                    table.add_row(change["rule_name"], str(old_priority), str(new_priority), impact)

                console.print(table)

                if len(results["changes"]) > 10:
                    remaining = len(results["changes"]) - 10
                    print_info(f"+ {remaining} more optimizations")

        else:
            print_success("Analysis complete")
            print_info("Rules are already well-optimized!", "No further improvements possible")

        # Handle output file
        if output:
            try:
                output.write_text(json.dumps(results, indent=2))
                print_success(f"Report saved to {output}")
            except Exception as e:
                error_msg = handle_error(e, debug=DEBUG_MODE)
                suggestion = get_error_suggestion(e)
                print_error(f"Failed to save report: {error_msg}", suggestion)

        # Next steps
        if not QUIET_MODE:
            console.print()
            if not apply and optimized > 0:
                console.print(
                    Panel(
                        "💡 Ready to apply? Run with [bold green]--apply[/bold green] flag",
                        border_style="green",
                        title="Next Steps",
                    )
                )
            elif apply and optimized > 0:
                console.print(
                    Panel(
                        f"🎉 Applied priority optimizations with [bold yellow]{improvement:.1f}% theoretical improvement![/bold yellow]\n"
                        "Note: Performance claims are unverified - actual results may vary.",
                        border_style="yellow",
                        title="Optimization Applied!",
                    )
                )

    except Exception as e:
        error_msg = handle_error(e, debug=DEBUG_MODE)
        suggestion = get_error_suggestion(e, "optimization")
        print_error(f"Optimization failed: {error_msg}", suggestion)
        sys.exit(1)


@cli.command()
@click.argument("project_path", type=click.Path(exists=True, path_type=Path), default=".")
def benchmark(project_path: Path):
    """📊 Benchmark optimization impact before applying changes."""

    if not QUIET_MODE:
        console.print()
        console.print(
            Panel(
                f"📊 Benchmarking [cyan]{project_path}[/cyan]",
                title="Performance Analysis",
                border_style="cyan",
            )
        )
        console.print()

    try:
        # Analysis with progress
        with create_progress_bar() as progress:
            task = progress.add_task("Analyzing performance impact...", total=100)

            optimizer = UnifiedOptimizer()
            progress.update(task, completed=50, description="Computing optimization potential...")

            results = optimizer.optimize(project_path, apply=False)
            progress.update(task, completed=100, description="Benchmark complete!")

        total_rules = results["total_rules"]
        optimizable = results["rules_changed"]
        improvement = results["estimated_improvement"]

        if total_rules == 0:
            print_warning("No simp rules found", "Ensure your project has @[simp] annotations")
            sys.exit(1)

        # Performance summary
        print_success(f"Performance analysis complete")
        print_info(f"Total simp rules: {total_rules}")
        print_info(f"Optimization candidates: {optimizable}")

        if optimizable > 0:
            print_success(f"Theoretical improvement: {improvement:.1f}% (unverified)")

            if VERBOSE_MODE:
                print_info(
                    "Priority optimizations available",
                    "Run 'simpulse optimize --apply' to apply changes (results not guaranteed)",
                )
        else:
            print_info("Rules are already optimized", "No performance improvements available")

        # Show hottest rules
        if results["changes"] and not QUIET_MODE:
            console.print()
            table = Table(title="🔥 High-Impact Rules", title_style="bold red")
            table.add_column("Rule", style="cyan")
            table.add_column("Current Priority", style="yellow", justify="center")
            table.add_column("Usage Frequency", style="green")
            table.add_column("Impact", style="red")

            for change in results["changes"][:5]:
                # Determine impact level
                old_priority = change["old_priority"]
                if old_priority >= 1000:
                    impact = "🚀 High"
                elif old_priority >= 500:
                    impact = "⚡ Medium"
                else:
                    impact = "✨ Low"

                table.add_row(change["rule_name"], str(old_priority), change["reason"], impact)

            console.print(table)

            if len(results["changes"]) > 5:
                remaining = len(results["changes"]) - 5
                print_info(f"+ {remaining} more optimization opportunities")

        # Recommendations
        if not QUIET_MODE and optimizable > 0:
            console.print()
            if improvement >= 10:
                recommendation = "⚡ High impact optimization available!"
            elif improvement >= 5:
                recommendation = "💡 Moderate improvement possible"
            else:
                recommendation = "✨ Small but measurable improvement"

            console.print(
                Panel(
                    f"{recommendation}\n"
                    f"Run [bold green]simpulse optimize --apply[/bold green] to optimize",
                    border_style="green",
                    title="Recommendation",
                )
            )

    except Exception as e:
        error_msg = handle_error(e, debug=DEBUG_MODE)
        suggestion = get_error_suggestion(e, "benchmark")
        print_error(f"Benchmark failed: {error_msg}", suggestion)
        sys.exit(1)


@cli.command()
def list_strategies():
    """📋 List available optimization strategies and their use cases."""

    if not QUIET_MODE:
        console.print()
        console.print(
            Panel(
                "Available optimization strategies for different use cases",
                title="📋 Optimization Strategies",
                border_style="cyan",
            )
        )
        console.print()

    strategies = [
        {
            "name": "frequency",
            "description": "Prioritize rules by usage frequency",
            "use_case": "Most projects - fastest proof search",
            "style": "green",
            "recommended": True,
        },
        {
            "name": "balanced",
            "description": "Balance frequency with complexity",
            "use_case": "Large projects with mixed complexity",
            "style": "yellow",
            "recommended": False,
        },
        {
            "name": "conservative",
            "description": "Only optimize very frequent rules",
            "use_case": "Stable projects - minimal changes",
            "style": "blue",
            "recommended": False,
        },
    ]

    if QUIET_MODE:
        # Just list names for scripting
        for strategy in strategies:
            console.print(strategy["name"])
        return

    # Beautiful table display
    table = Table(title="✨ Choose Your Strategy", title_style="bold cyan")
    table.add_column("Strategy", style="bold")
    table.add_column("Description", style="dim")
    table.add_column("Best For", style="italic")
    table.add_column("Status", justify="center")

    for strategy in strategies:
        status = "⭐ Recommended" if strategy["recommended"] else ""

        table.add_row(
            f"[{strategy['style']}]{strategy['name']}[/{strategy['style']}]",
            strategy["description"],
            strategy["use_case"],
            status,
        )

    console.print(table)
    console.print()

    # Usage examples
    if VERBOSE_MODE:
        print_info("Usage examples:")
        console.print("  [dim]# Use recommended strategy[/dim]")
        console.print("  [green]simpulse optimize[/green]")
        console.print()
        console.print("  [dim]# Specify strategy[/dim]")
        console.print("  [green]simpulse optimize --strategy conservative[/green]")
        console.print()
        console.print("  [dim]# Preview before applying[/dim]")
        console.print("  [green]simpulse optimize --strategy frequency[/green]")
        console.print("  [green]simpulse optimize --strategy frequency --apply[/green]")
    else:
        console.print(
            Panel(
                "Use with: [bold green]simpulse optimize --strategy <name>[/bold green]",
                border_style="green",
                title="Usage",
            )
        )


@cli.command()
@click.argument("project_path", type=click.Path(exists=True))
@click.option("--save-prediction", is_flag=True, help="Save prediction for later verification")
def guarantee(project_path, save_prediction):
    """Analyze optimization potential and provide performance guarantee."""

    with handle_error("performance guarantee analysis"):
        optimizer = UnifiedOptimizer()
        guarantee_system = PerformanceGuarantee()

        console.print(
            Panel.fit(
                "🎯 Simpulse Performance Guarantee",
                subtitle="Honest assessment of optimization potential",
            )
        )

        with create_progress_bar() as progress:
            task = progress.add_task("Analyzing project...", total=100)

            # Analyze project
            progress.update(task, advance=50, description="Finding simp rules...")
            results = optimizer.optimize(project_path, apply=False)

            progress.update(task, advance=30, description="Analyzing usage patterns...")
            prediction = guarantee_system.analyze_optimization_potential(results)

            progress.update(task, advance=20, description="Generating report...")

        # Display comprehensive report
        report = guarantee_system.format_prediction_report(prediction)
        console.print(report)

        # Show accuracy statistics
        accuracy_stats = guarantee_system.get_prediction_accuracy()
        if accuracy_stats["total_predictions"] > 0:
            console.print(
                f"\n📊 Historical accuracy: {accuracy_stats['accuracy']:.1%} "
                f"({accuracy_stats['verified_predictions']}/{accuracy_stats['total_predictions']} predictions verified)"
            )

        # Exit with appropriate code for scripting
        if prediction.recommendation == "optimize":
            sys.exit(0)  # Success - should optimize
        elif prediction.recommendation == "maybe":
            sys.exit(1)  # Caution - test first
        else:
            sys.exit(2)  # Skip - won't help


def main():
    """Main entry point."""
    # Handle --health before Click processes commands
    if "--health" in sys.argv:
        try:
            import tempfile

            from .config import LEAN_PATH
            from .unified_optimizer import UnifiedOptimizer

            console_health = Console()

            # Test 1: Can create optimizer
            optimizer = UnifiedOptimizer()

            # Test 2: Can process a simple project
            with tempfile.TemporaryDirectory() as temp_dir:
                test_file = Path(temp_dir) / "test.lean"
                test_file.write_text("@[simp] theorem test : 1 = 1 := by simp")

                optimizer.optimize(temp_dir)

            # Test 3: Check configuration
            console_health.print("✅ Health check passed", style="green")
            console_health.print(f"  - Optimizer: OK")
            console_health.print(f"  - File processing: OK")
            console_health.print(f"  - Lean path: {LEAN_PATH}")
            sys.exit(0)

        except Exception as e:
            console_health = Console()
            console_health.print("❌ Health check failed", style="red")
            console_health.print(f"  Error: {e}")
            sys.exit(1)

    cli()


if __name__ == "__main__":
    main()
