"""Enhanced CLI for Simpulse with production features.

This module provides a comprehensive command-line interface for
Simpulse optimization with GitHub integration, monitoring, and reporting.
"""

import asyncio
import json
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from .config import Config, load_config
from .evolution.evolution_engine import EvolutionEngine
from .deployment.github_action import GitHubActionRunner
from .reporting.report_generator import ReportGenerator
from .monitoring.metrics_collector import MetricsCollector

console = Console()


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None):
    """Setup rich logging with optional file output."""
    handlers = [RichHandler(console=console, rich_tracebacks=True)]
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers
    )


@click.group()
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--log-file', type=click.Path(), help='Log file path')
@click.pass_context
def main(ctx, config, log_level, log_file):
    """🧬 Simpulse - Evolutionary Simp Optimization for Lean 4."""
    ctx.ensure_object(dict)
    
    # Setup logging
    setup_logging(log_level, Path(log_file) if log_file else None)
    
    # Load configuration
    ctx.obj['config'] = load_config(config)
    ctx.obj['config'].setup_logging()
    
    # Display banner
    if ctx.invoked_subcommand != 'version':
        console.print(Panel(
            "[bold green]🧬 Simpulse Optimizer[/bold green]\n"
            "Evolutionary optimization for Lean 4 simp tactics",
            expand=False
        ))


@main.command()
@click.option('--modules', help='Modules to optimize (comma-separated or "auto")')
@click.option('--time-budget', type=int, default=7200, help='Time budget in seconds')
@click.option('--target-improvement', type=float, default=15.0, help='Target improvement %')
@click.option('--population-size', type=int, default=30, help='Population size')
@click.option('--max-generations', type=int, default=50, help='Maximum generations')
@click.option('--parallel-workers', type=int, default=4, help='Parallel workers')
@click.option('--claude-backend', type=click.Choice(['claude_code', 'api']), default='claude_code')
@click.option('--create-pr/--no-create-pr', default=False, help='Create GitHub PR')
@click.option('--pr-branch', help='PR branch name')
@click.option('--base-branch', default='main', help='Base branch for PR')
@click.option('--cache-dir', type=click.Path(), help='Cache directory')
@click.option('--output-dir', type=click.Path(), help='Output directory')
@click.option('--progress-comments/--no-progress-comments', default=False, help='Post progress comments')
@click.option('--report-format', type=click.Choice(['html', 'markdown', 'both']), default='both')
@click.option('--enable-telemetry/--disable-telemetry', default=True, help='Enable telemetry')
@click.option('--dry-run/--no-dry-run', default=False, help='Dry run mode')
@click.pass_context
async def optimize(ctx, modules, time_budget, target_improvement, population_size, 
                  max_generations, parallel_workers, claude_backend, create_pr, 
                  pr_branch, base_branch, cache_dir, output_dir, progress_comments,
                  report_format, enable_telemetry, dry_run):
    """Run evolutionary optimization on Lean simp rules."""
    
    config = ctx.obj['config']
    
    # Parse modules
    if modules == "auto":
        module_list = await _auto_detect_modules()
    else:
        module_list = [m.strip() for m in modules.split(',') if m.strip()]
    
    if not module_list:
        console.print("[red]❌ No modules specified or detected[/red]")
        sys.exit(1)
    
    # Setup directories
    output_path = Path(output_dir) if output_dir else config.paths.output_dir
    cache_path = Path(cache_dir) if cache_dir else config.paths.cache_dir
    
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[green]🎯 Optimizing modules:[/green] {', '.join(module_list)}")
    console.print(f"[blue]⚙️ Configuration:[/blue]")
    console.print(f"  Time budget: {time_budget}s")
    console.print(f"  Target improvement: {target_improvement}%")
    console.print(f"  Population: {population_size}")
    console.print(f"  Max generations: {max_generations}")
    console.print(f"  Workers: {parallel_workers}")
    console.print(f"  Output: {output_path}")
    
    if dry_run:
        console.print("[yellow]🧪 Running in dry-run mode[/yellow]")
    
    # Initialize components
    run_id = str(uuid.uuid4())[:8]
    
    try:
        # Setup metrics collection
        metrics_collector = MetricsCollector(
            storage_dir=output_path / "metrics",
            enable_telemetry=enable_telemetry
        )
        
        # Setup GitHub integration
        github_runner = None
        if create_pr:
            github_runner = GitHubActionRunner(dry_run=dry_run)
            validation = github_runner.validate_github_connection()
            
            if validation["status"] == "error":
                console.print(f"[red]❌ GitHub setup failed:[/red] {validation['message']}")
                if not dry_run:
                    sys.exit(1)
            else:
                console.print("[green]✓ GitHub integration ready[/green]")
        
        # Setup evolution engine
        evolution_engine = EvolutionEngine(config)
        
        # Start metrics tracking
        await metrics_collector.track_optimization_run(
            run_id, module_list, {
                "population_size": population_size,
                "max_generations": max_generations,
                "time_budget": time_budget,
                "target_improvement": target_improvement
            }
        )
        
        # Run optimization with progress tracking
        console.print("\n[bold]🚀 Starting optimization...[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            # Create progress task
            task = progress.add_task("Optimizing...", total=max_generations)
            
            # Run evolution
            result = await evolution_engine.run_evolution(
                modules=module_list,
                source_path=Path.cwd(),
                time_budget=time_budget
            )
            
            # Update progress to completion
            progress.update(task, completed=max_generations)
        
        # Complete metrics tracking
        await metrics_collector.complete_optimization_run(run_id, result)
        
        # Display results
        _display_results(result)
        
        # Generate reports
        report_generator = ReportGenerator()
        
        if report_format in ['html', 'both']:
            html_report = await report_generator.generate_html_report(result)
            html_path = output_path / "optimization_report.html"
            with open(html_path, 'w') as f:
                f.write(html_report)
            console.print(f"[green]📊 HTML report:[/green] {html_path}")
        
        if report_format in ['markdown', 'both']:
            md_report = report_generator.generate_markdown_summary(result)
            md_path = output_path / "optimization_report.md"
            with open(md_path, 'w') as f:
                f.write(md_report)
            console.print(f"[green]📝 Markdown report:[/green] {md_path}")
        
        # Save results JSON
        results_data = {
            "run_id": run_id,
            "improvement_percent": result.improvement_percent,
            "total_generations": result.total_generations,
            "execution_time": result.execution_time,
            "success": result.success,
            "modules": result.modules,
            "timestamp": datetime.now().isoformat()
        }
        
        if result.best_candidate:
            results_data["best_candidate"] = {
                "mutations": len(result.best_candidate.mutations),
                "fitness": result.best_candidate.fitness.composite_score if result.best_candidate.fitness else 0
            }
        
        results_path = output_path / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Create GitHub PR if requested
        pr_url = None
        if create_pr and github_runner and result.success:
            console.print("\n[blue]📤 Creating GitHub pull request...[/blue]")
            pr_url = await github_runner.create_optimization_pr(
                result, target_branch=pr_branch, source_branch=base_branch
            )
            
            if pr_url:
                console.print(f"[green]✓ Pull request created:[/green] {pr_url}")
                results_data["pr_url"] = pr_url
                
                # Update results file
                with open(results_path, 'w') as f:
                    json.dump(results_data, f, indent=2)
            else:
                console.print("[yellow]⚠️ Failed to create pull request[/yellow]")
        
        # Send telemetry
        if enable_telemetry:
            await metrics_collector.send_telemetry(anonymous=True)
        
        console.print(f"\n[bold green]🎉 Optimization completed successfully![/bold green]")
        console.print(f"[green]📈 Improvement:[/green] {result.improvement_percent:.1f}%")
        
        if result.success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Optimization interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]❌ Optimization failed:[/red] {str(e)}")
        logging.exception("Optimization failed")
        sys.exit(1)


@main.command()
@click.option('--daemon/--no-daemon', default=False, help='Run as background service')
@click.option('--webhook-url', help='Progress webhook URL')
@click.option('--cache-dir', type=click.Path(), help='Cache directory')
@click.option('--port', type=int, default=8080, help='Service port')
@click.pass_context
def serve(ctx, daemon, webhook_url, cache_dir, port):
    """Run Simpulse as a background service."""
    
    console.print("[blue]🚀 Starting Simpulse service...[/blue]")
    
    if daemon:
        console.print(f"[green]✓ Service mode enabled on port {port}[/green]")
        # In a real implementation, this would start a web service
        console.print("[yellow]⚠️ Service mode not yet implemented[/yellow]")
    else:
        console.print("[yellow]ℹ️ Interactive service mode[/yellow]")
        console.print("Press Ctrl+C to stop the service")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]🛑 Service stopped[/yellow]")


@main.command()
@click.option('--format', type=click.Choice(['html', 'markdown', 'json']), default='html')
@click.option('--include-graphs/--no-include-graphs', default=True, help='Include graphs')
@click.option('--input-file', type=click.Path(exists=True), help='Input results file')
@click.option('--output-file', type=click.Path(), help='Output report file')
@click.pass_context
def report(ctx, format, include_graphs, input_file, output_file):
    """Generate optimization report from results."""
    
    console.print(f"[blue]📊 Generating {format.upper()} report...[/blue]")
    
    if not input_file:
        console.print("[red]❌ Input file required[/red]")
        sys.exit(1)
    
    # Load results (simplified for demo)
    try:
        with open(input_file, 'r') as f:
            results_data = json.load(f)
        
        console.print(f"[green]✓ Loaded results:[/green] {results_data.get('run_id', 'unknown')}")
        
        # Generate mock report
        if format == 'html':
            report_content = f"""<!DOCTYPE html>
<html>
<head><title>Simpulse Report</title></head>
<body>
<h1>Optimization Report</h1>
<p>Improvement: {results_data.get('improvement_percent', 0):.1f}%</p>
<p>Execution time: {results_data.get('execution_time', 0):.1f}s</p>
</body>
</html>"""
        elif format == 'markdown':
            report_content = f"""# Optimization Report

- **Improvement**: {results_data.get('improvement_percent', 0):.1f}%
- **Execution time**: {results_data.get('execution_time', 0):.1f}s
- **Success**: {results_data.get('success', False)}
"""
        else:  # json
            report_content = json.dumps(results_data, indent=2)
        
        # Save report
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            console.print(f"[green]✓ Report saved:[/green] {output_file}")
        else:
            console.print(report_content)
            
    except Exception as e:
        console.print(f"[red]❌ Report generation failed:[/red] {str(e)}")
        sys.exit(1)


@main.command()
@click.option('--config-file', type=click.Path(), help='Configuration file to validate')
@click.pass_context
def validate(ctx, config_file):
    """Validate configuration and environment."""
    
    console.print("[blue]🔍 Validating Simpulse environment...[/blue]")
    
    config = ctx.obj['config']
    
    # Validation table
    table = Table(title="Environment Validation")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="green")
    
    # Check Python environment
    table.add_row("Python", "✓", f"{sys.version.split()[0]}")
    
    # Check Lean installation
    try:
        import subprocess
        result = subprocess.run(['lean', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            table.add_row("Lean 4", "✓", result.stdout.strip())
        else:
            table.add_row("Lean 4", "❌", "Not found")
    except:
        table.add_row("Lean 4", "❌", "Not found")
    
    # Check Lake
    try:
        result = subprocess.run(['lake', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            table.add_row("Lake", "✓", result.stdout.strip())
        else:
            table.add_row("Lake", "❌", "Not found")
    except:
        table.add_row("Lake", "❌", "Not found")
    
    # Check Claude Code CLI
    try:
        result = subprocess.run(['claude', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            table.add_row("Claude Code", "✓", "Available")
        else:
            table.add_row("Claude Code", "⚠️", "Not available")
    except:
        table.add_row("Claude Code", "⚠️", "Not available")
    
    # Check GitHub integration
    github_runner = GitHubActionRunner(dry_run=True)
    validation = github_runner.validate_github_connection()
    
    if validation["status"] == "success":
        table.add_row("GitHub", "✓", f"Connected to {validation.get('repository', 'unknown')}")
    else:
        table.add_row("GitHub", "⚠️", validation["message"])
    
    # Check directories
    for name, path in [
        ("Output Dir", config.paths.output_dir),
        ("Cache Dir", config.paths.cache_dir),
        ("Log Dir", config.paths.log_dir)
    ]:
        if path.exists():
            table.add_row(name, "✓", str(path))
        else:
            table.add_row(name, "⚠️", f"Will be created: {path}")
    
    console.print(table)
    
    # Configuration summary
    console.print(f"\n[blue]📋 Configuration:[/blue]")
    console.print(f"  Population size: {config.optimization.population_size}")
    console.print(f"  Max generations: {config.optimization.generations}")
    console.print(f"  Claude backend: {config.claude.backend.value}")
    console.print(f"  Parallel workers: {config.optimization.max_parallel_evaluations}")


@main.command()
def version():
    """Show version information."""
    console.print("[bold]Simpulse Evolutionary Optimizer[/bold]")
    console.print("Version: 0.1.0")
    console.print("Author: Simpulse Team")


def _display_results(result):
    """Display optimization results in a nice format."""
    
    # Results panel
    if result.success:
        panel_style = "green"
        status_icon = "🎉"
        status_text = "SUCCESS"
    else:
        panel_style = "red"
        status_icon = "❌"
        status_text = "FAILED"
    
    console.print(f"\n[bold {panel_style}]{status_icon} Optimization {status_text}[/bold {panel_style}]")
    
    # Results table
    table = Table(title="Optimization Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Improvement", f"{result.improvement_percent:.1f}%")
    table.add_row("Generations", str(result.total_generations))
    table.add_row("Evaluations", str(result.total_evaluations))
    table.add_row("Execution Time", f"{result.execution_time:.1f}s")
    table.add_row("Modules", str(len(result.modules)))
    
    if result.best_candidate:
        table.add_row("Mutations Applied", str(len(result.best_candidate.mutations)))
        if result.best_candidate.fitness:
            table.add_row("Best Fitness", f"{result.best_candidate.fitness.composite_score:.4f}")
    
    console.print(table)


async def _auto_detect_modules() -> List[str]:
    """Auto-detect Lean modules in the current directory."""
    modules = []
    
    # Look for .lean files
    for lean_file in Path.cwd().rglob("*.lean"):
        # Convert file path to module name
        relative_path = lean_file.relative_to(Path.cwd())
        
        # Skip if in build directories
        if any(part.startswith('.') for part in relative_path.parts):
            continue
        if 'build' in relative_path.parts:
            continue
            
        module_name = str(relative_path.with_suffix('')).replace('/', '.')
        modules.append(module_name)
    
    return sorted(list(set(modules))[:10])  # Limit to 10 modules


def cli_main():
    """Main CLI entry point for async support."""
    import sys
    
    # Check if we're running an async command
    async_commands = ['optimize']
    
    if len(sys.argv) > 1 and sys.argv[1] in async_commands:
        # Run async command
        @main.command()
        @click.pass_context
        def async_wrapper(ctx):
            # Get the actual command
            cmd_name = sys.argv[1]
            cmd = main.get_command(ctx, cmd_name)
            
            # Create new context for the command
            with main.make_context(cmd_name, sys.argv[2:], parent=ctx) as cmd_ctx:
                # Run the async command
                return asyncio.run(cmd.callback(**cmd_ctx.params))
        
        # Replace the command temporarily
        original_cmd = main.commands[sys.argv[1]]
        main.commands[sys.argv[1]] = async_wrapper
        
        try:
            main()
        finally:
            main.commands[sys.argv[1]] = original_cmd
    else:
        main()


if __name__ == '__main__':
    cli_main()