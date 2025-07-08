"""Simple, direct CLI for Simpulse - no frameworks, just functionality."""

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .unified_optimizer import UnifiedOptimizer

console = Console()


@click.group()
@click.version_option(version="2.0.0", prog_name="simpulse")
def cli():
    """Simpulse - Simple optimizer for Lean 4 simp rules."""


@cli.command()
@click.argument("project_path", type=click.Path(exists=True, path_type=Path), default=".")
def check(project_path: Path):
    """Check if your project could benefit from optimization."""
    console.print(f"\n🔍 Checking [cyan]{project_path}[/cyan]...\n")

    # Quick analysis
    optimizer = UnifiedOptimizer()
    results = optimizer.optimize(project_path, apply=False)

    total_rules = results["total_rules"]
    optimizable = results["rules_changed"]

    if optimizable > 0:
        console.print(f"✅ Found [green]{total_rules}[/green] simp rules")
        console.print(f"💡 Can optimize [yellow]{optimizable}[/yellow] rules")
        console.print(
            f"🚀 Potential improvement: [cyan]{results['estimated_improvement']:.1f}%[/cyan]"
        )
        console.print(f"\nRun [bold]simpulse optimize[/bold] to optimize your project")
    else:
        console.print(f"✅ Found [green]{total_rules}[/green] simp rules")
        console.print("✨ Your simp rules are already well-optimized!")


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
    """Optimize simp rule priorities."""
    if not as_json:
        console.print(
            f"\n🧠 Optimizing [cyan]{project_path}[/cyan] with [yellow]{strategy}[/yellow] strategy...\n"
        )

    # Run optimization
    optimizer = UnifiedOptimizer(strategy=strategy)
    results = optimizer.optimize(project_path, apply=apply)

    if as_json:
        # JSON output
        click.echo(json.dumps(results, indent=2))
    else:
        # Human-readable output
        console.print("✨ Optimization complete!")
        console.print(f"   Total rules: [cyan]{results['total_rules']}[/cyan]")
        console.print(f"   Rules optimized: [green]{results['rules_changed']}[/green]")
        console.print(
            f"   Estimated improvement: [yellow]{results['estimated_improvement']:.1f}%[/yellow]\n"
        )

        if results["changes"]:
            # Show table of changes
            table = Table(title="Optimization Changes (first 10)")
            table.add_column("Rule", style="cyan")
            table.add_column("Current", style="red")
            table.add_column("Optimized", style="green")
            table.add_column("Reason", style="yellow")

            for change in results["changes"][:10]:
                table.add_row(
                    change["rule_name"],
                    str(change["old_priority"]),
                    str(change["new_priority"]),
                    change["reason"],
                )

            console.print(table)

            if len(results["changes"]) > 10:
                console.print(f"\n... and {len(results['changes']) - 10} more changes")

        if output:
            # Save to file
            output.write_text(json.dumps(results, indent=2))
            console.print(f"\n💾 Optimization plan saved to: [cyan]{output}[/cyan]")
        elif not apply and results["changes"]:
            console.print(f"\nRun with [bold]--apply[/bold] to apply these changes")
        elif apply:
            console.print(f"\n✅ Changes applied successfully!")


@cli.command()
@click.argument("project_path", type=click.Path(exists=True, path_type=Path), default=".")
def benchmark(project_path: Path):
    """Simple benchmark of optimization impact."""
    console.print(f"\n⚡ Benchmarking [cyan]{project_path}[/cyan]...\n")

    # For now, just show what would be optimized
    optimizer = UnifiedOptimizer()
    results = optimizer.optimize(project_path, apply=False)

    console.print("📊 Benchmark Summary:")
    console.print(f"   Total simp rules: [cyan]{results['total_rules']}[/cyan]")
    console.print(f"   Frequently used rules: [green]{results['rules_changed']}[/green]")
    console.print(f"   Expected speedup: [yellow]{results['estimated_improvement']:.1f}%[/yellow]")

    if results["changes"]:
        console.print(f"\n🔥 Hottest rules (most frequently used):")
        for change in results["changes"][:5]:
            console.print(f"   • {change['rule_name']} ({change['reason']})")


@cli.command()
def list_strategies():
    """List available optimization strategies."""
    console.print("\n📋 Available Optimization Strategies:\n")

    strategies = {
        "frequency": "Prioritize rules by usage frequency (recommended)",
        "balanced": "Balance frequency with complexity considerations",
        "conservative": "Only optimize very frequently used rules",
    }

    for name, description in strategies.items():
        console.print(f"  [cyan]{name}[/cyan]: {description}")

    console.print(f"\nUse with: [bold]simpulse optimize --strategy <name>[/bold]")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
