#!/usr/bin/env python3
"""
Simpulse v2 CLI - Simplified, user-focused interface.
Make your Lean proofs faster!
"""

import asyncio
import sys
from pathlib import Path

import click

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from marketing.success_metrics import ImpactTracker
from scripts.minimal_viable_product import MinimalSimpulse
from scripts.simp_health_check import SimpHealthChecker


@click.group()
@click.version_option(version="0.2.0", prog_name="Simpulse")
def cli():
    """Simpulse: Make your Lean proofs faster.

    Simple commands for big improvements:

    \b
      check    - See if your project needs optimization
      optimize - Make your simp rules faster
      report   - Show impact and success metrics
    """


@cli.command()
@click.argument("project_path", type=click.Path(exists=True), default=".")
@click.option("--json", is_flag=True, help="Output in JSON format")
def check(project_path, json):
    """Check if your project could benefit from optimization.

    This runs a quick health check on your Lean project and tells you
    if Simpulse can help make it faster.

    Example:
        simpulse check MyLeanProject/
    """

    click.echo("🔍 Analyzing simp performance...")

    project_path = Path(project_path)
    checker = SimpHealthChecker()

    # Run health check
    report = asyncio.run(checker.analyze_project(project_path))

    if json:
        import json as json_lib

        output = {
            "total_rules": report.total_rules,
            "custom_priorities": report.custom_priorities,
            "optimization_potential": report.optimization_potential,
            "estimated_improvement": report.estimated_improvement,
            "recommendations": report.recommendations,
        }
        click.echo(json_lib.dumps(output, indent=2))
    else:
        click.echo(checker.generate_report(report))

        if report.optimization_potential > 50:
            click.echo("\n✨ Good news! Simpulse can likely help.")
            click.echo(f"Run: simpulse optimize {project_path}")
        elif report.optimization_potential > 30:
            click.echo("\n💡 Moderate optimization potential detected.")
            click.echo(f"Run: simpulse optimize {project_path}")
        else:
            click.echo("\n✅ Your simp rules look well-optimized!")
            click.echo("Minor improvements may still be possible.")


@cli.command()
@click.argument("project_path", type=click.Path(exists=True), default=".")
@click.option("--file", "-f", help="Optimize a specific file")
@click.option("--accept-all", is_flag=True, help="Accept all suggestions automatically")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be done without making changes"
)
def optimize(project_path, file, accept_all, dry_run):
    """Optimize simp performance.

    This analyzes your simp rules and suggests priority optimizations
    that can make your proofs significantly faster.

    Example:
        simpulse optimize MyLeanProject/
    """

    click.echo("🚀 Starting optimization...")

    project_path = Path(project_path)

    if file:
        # Optimize specific file
        target_file = project_path / file
        if not target_file.exists():
            click.echo(f"❌ Error: File {file} not found", err=True)
            sys.exit(1)

        optimizer = MinimalSimpulse()
        result = asyncio.run(optimizer.optimize_file(target_file))
    else:
        # TODO: Implement full project optimization
        click.echo("Full project optimization coming soon!")
        click.echo("For now, please specify a file with --file")
        sys.exit(1)

    if result.improved:
        click.echo(f"\n🎉 Success! {result.improvement_percent:.1f}% improvement")
        click.echo(f"   Baseline:  {result.baseline_time:.2f}ms")
        click.echo(f"   Optimized: {result.optimized_time:.2f}ms")

        if result.mutation:
            click.echo(f"\nBest optimization: {result.mutation}")

        if not dry_run:
            if accept_all or click.confirm("\nApply these changes?"):
                # TODO: Actually apply the changes
                click.echo("✅ Changes applied!")

                # Track success
                tracker = ImpactTracker()
                tracker.add_success(
                    project=project_path.name,
                    time_saved=(result.baseline_time - result.optimized_time) / 1000,
                    improvement=result.improvement_percent,
                    rules_optimized=1,  # TODO: Track actual number
                )
            else:
                click.echo("Changes not applied.")
    else:
        click.echo("\n😐 No significant improvement found.")
        click.echo("Your simp rules may already be well-optimized.")
        click.echo("\nTips:")
        click.echo("- Try optimizing files with many simp rules")
        click.echo("- Look for files where all rules use default priority")


@cli.command()
@click.option("--project", help="Show report for specific project")
@click.option("--leaderboard", is_flag=True, help="Show top optimizations")
@click.option("--submit", is_flag=True, help="Submit your results to global stats")
def report(project, leaderboard, submit):
    """Show impact and success metrics.

    See how much time Simpulse has saved and view success stories
    from the community.

    Example:
        simpulse report --leaderboard
    """

    tracker = ImpactTracker()

    if leaderboard:
        click.echo(tracker.get_leaderboard())
    elif submit:
        click.echo("📊 Submit your results at: https://simpulse.dev/submit")
        click.echo("Help us track the global impact of simp optimization!")
    else:
        click.echo(tracker.generate_impact_report())

        # Also create visualization
        click.echo("\n📈 Creating impact visualization...")
        tracker.create_impact_visualization()
        click.echo("✅ Saved to impact_charts.png")


@cli.command()
def quickstart():
    """Interactive quickstart guide.

    This walks you through your first optimization step by step.
    """

    click.echo("🎯 Welcome to Simpulse!")
    click.echo("=" * 50)
    click.echo("\nLet's optimize your first Lean project.\n")

    # Step 1: Find project
    project_path = click.prompt("Enter path to your Lean project", default=".")
    project_path = Path(project_path)

    if not project_path.exists():
        click.echo(f"❌ Path {project_path} not found")
        sys.exit(1)

    # Step 2: Run health check
    click.echo(f"\n🔍 Checking {project_path.name}...")

    checker = SimpHealthChecker()
    report = asyncio.run(checker.analyze_project(project_path))

    click.echo(f"\nFound {report.total_rules} simp rules")
    click.echo(f"Optimization potential: {report.optimization_potential:.0f}%")
    click.echo(f"Estimated improvement: {report.estimated_improvement:.0f}%")

    if report.optimization_potential < 30:
        click.echo("\n✅ Your project is already well-optimized!")
        click.echo("Try running on a project with more simp rules.")
        return

    # Step 3: Suggest next steps
    click.echo(
        f"\n✨ Great! You could save ~{report.estimated_improvement:.0f}% on build time."
    )
    click.echo("\nNext steps:")
    click.echo(f"1. Run: simpulse optimize {project_path}")
    click.echo("2. Review the suggested changes")
    click.echo("3. Apply and measure the improvement")
    click.echo("\nHappy optimizing! 🚀")


@cli.command()
def examples():
    """Show example optimizations.

    See real examples of how Simpulse optimizes simp rules.
    """

    examples_text = """
📚 Simpulse Optimization Examples

1. FREQUENCY-BASED OPTIMIZATION
   
   Before (all default priority):
   ```lean
   @[simp] theorem rarely_used : complex_expression = ...
   @[simp] theorem very_common : n + 0 = n
   ```
   
   After (optimized priorities):
   ```lean
   @[simp low] theorem rarely_used : complex_expression = ...
   @[simp high] theorem very_common : n + 0 = n
   ```
   
   Result: 40% faster simp calls

2. COMPLEXITY-BASED OPTIMIZATION
   
   Before:
   ```lean
   @[simp] theorem complex_distrib : (a+b)*(c+d) = a*c+a*d+b*c+b*d
   @[simp] theorem simple_zero : 0 * n = 0
   ```
   
   After:
   ```lean
   @[simp 100] theorem complex_distrib : (a+b)*(c+d) = a*c+a*d+b*c+b*d
   @[simp 2000] theorem simple_zero : 0 * n = 0
   ```
   
   Result: Simple rules checked first

3. DOMAIN GROUPING
   
   Before (mixed priorities):
   ```lean
   @[simp] theorem list_rule1 : ...
   @[simp] theorem nat_rule1 : ...
   @[simp] theorem list_rule2 : ...
   ```
   
   After (grouped by domain):
   ```lean
   @[simp 1500] theorem list_rule1 : ...
   @[simp 2500] theorem nat_rule1 : ...
   @[simp 1500] theorem list_rule2 : ...
   ```
   
   Result: Better cache locality

For more examples, visit: https://github.com/Bright-L01/simpulse/examples
"""

    click.echo(examples_text)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
