#!/usr/bin/env python3
"""
Advanced Simpulse Features Example

This script demonstrates advanced features including:
- Custom fitness functions
- GitHub integration
- Continuous optimization
- Metrics collection and reporting
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

from simpulse.config import Config, OptimizationConfig, ClaudeConfig, PathConfig
from simpulse.evolution.evolution_engine import EvolutionEngine
from simpulse.deployment.github_action import GitHubActionRunner
from simpulse.deployment.continuous_optimizer import ContinuousOptimizer
from simpulse.monitoring.metrics_collector import MetricsCollector
from simpulse.reporting.report_generator import ReportGenerator


async def github_integration_example():
    """Example showing GitHub integration for automated PRs."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🐙 GitHub Integration Example")
    
    # Setup configuration
    config = Config(
        optimization=OptimizationConfig(
            population_size=25,
            generations=30,
            time_budget=2400,  # 40 minutes
            target_improvement=12.0
        ),
        claude=ClaudeConfig(backend="claude_code"),
        paths=PathConfig(
            output_dir=Path("./github_output"),
            cache_dir=Path("./github_cache"),
            log_dir=Path("./github_logs")
        )
    )
    
    # Create directories
    for path in [config.paths.output_dir, config.paths.cache_dir, config.paths.log_dir]:
        path.mkdir(exist_ok=True)
    
    # Initialize components
    engine = EvolutionEngine(config)
    github_runner = GitHubActionRunner(dry_run=False)  # Set to True for testing
    
    # Validate GitHub connection
    validation = github_runner.validate_github_connection()
    if validation["status"] != "success":
        logger.error(f"GitHub validation failed: {validation['message']}")
        return
    
    logger.info("✅ GitHub connection validated")
    
    # Run optimization
    modules = ["MyProject.Core", "MyProject.Utils"]
    
    try:
        result = await engine.run_evolution(
            modules=modules,
            source_path=Path.cwd(),
            time_budget=config.optimization.time_budget
        )
        
        if result.success and result.improvement_percent > 5.0:
            logger.info("🚀 Creating GitHub PR for optimization results...")
            
            pr_url = await github_runner.create_optimization_pr(
                result,
                target_branch="simpulse/auto-optimization",
                source_branch="main"
            )
            
            if pr_url:
                logger.info(f"✅ Pull request created: {pr_url}")
            else:
                logger.warning("⚠️ Failed to create pull request")
        else:
            logger.info("📊 No significant improvement found, skipping PR creation")
            
    except Exception as e:
        logger.error(f"❌ GitHub integration example failed: {e}")


async def continuous_optimization_example():
    """Example of setting up continuous optimization service."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("⚡ Continuous Optimization Example")
    
    # Setup configuration
    config = Config(
        optimization=OptimizationConfig(
            population_size=20,
            generations=25,
            time_budget=1800,
            target_improvement=8.0,
            max_parallel_evaluations=2
        ),
        claude=ClaudeConfig(backend="claude_code"),
        paths=PathConfig(
            output_dir=Path("./continuous_output"),
            cache_dir=Path("./continuous_cache"),
            log_dir=Path("./continuous_logs")
        )
    )
    
    # Create directories
    for path in [config.paths.output_dir, config.paths.cache_dir, config.paths.log_dir]:
        path.mkdir(exist_ok=True)
    
    # Initialize continuous optimizer
    optimizer = ContinuousOptimizer(config)
    
    try:
        # Start the service
        await optimizer.start_service()
        logger.info("✅ Continuous optimization service started")
        
        # Schedule weekly optimization
        success = await optimizer.schedule_optimization(
            trigger_id="weekly_optimization",
            modules=["MyProject.Core", "MyProject.Algorithms"],
            cron_expression="0 2 * * 0",  # Sunday 2 AM
            config_overrides={"create_pr": True}
        )
        
        if success:
            logger.info("📅 Weekly optimization scheduled")
        
        # Setup commit hook for specific paths
        await optimizer.setup_commit_hook(
            trigger_id="lean_changes",
            modules=["auto"],  # Auto-detect modules
            watched_paths=["src/", "lib/"],
            branch_patterns=["main", "develop"]
        )
        logger.info("🪝 Commit hook configured")
        
        # Trigger a manual optimization for demonstration
        run_id = await optimizer.trigger_manual_optimization(
            modules=["MyProject.Example"],
            config_overrides={"time_budget": 600}  # 10 minutes for demo
        )
        
        if run_id:
            logger.info(f"🎯 Manual optimization triggered: {run_id}")
            
            # Monitor the run
            for i in range(30):  # Check for 5 minutes
                await asyncio.sleep(10)
                status = optimizer.get_optimization_status(run_id)
                
                if status:
                    logger.info(f"📊 Run {run_id}: {status['status']}")
                    
                    if status['status'] in ['completed', 'failed', 'cancelled']:
                        if status['status'] == 'completed':
                            improvement = status.get('improvement_percent', 0)
                            logger.info(f"✅ Optimization completed: {improvement:.1f}% improvement")
                        break
                else:
                    logger.warning(f"⚠️ Could not get status for run {run_id}")
                    break
        
        # Show service status
        service_status = optimizer.get_service_status()
        logger.info(f"📈 Service status: {service_status}")
        
        # List all triggers
        triggers = optimizer.list_triggers()
        logger.info(f"🔧 Active triggers: {len(triggers)}")
        for trigger in triggers:
            logger.info(f"  - {trigger['trigger_id']}: {trigger['trigger_type']} ({'enabled' if trigger['enabled'] else 'disabled'})")
        
    except Exception as e:
        logger.error(f"❌ Continuous optimization example failed: {e}")
    finally:
        # Stop the service
        await optimizer.stop_service()
        logger.info("🛑 Continuous optimization service stopped")


async def metrics_and_reporting_example():
    """Example showing metrics collection and report generation."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("📊 Metrics and Reporting Example")
    
    # Setup configuration
    config = Config(
        optimization=OptimizationConfig(
            population_size=15,
            generations=20,
            time_budget=1200,
            target_improvement=8.0
        ),
        claude=ClaudeConfig(backend="claude_code"),
        paths=PathConfig(
            output_dir=Path("./metrics_output"),
            cache_dir=Path("./metrics_cache"),
            log_dir=Path("./metrics_logs")
        )
    )
    
    # Create directories
    for path in [config.paths.output_dir, config.paths.cache_dir, config.paths.log_dir]:
        path.mkdir(exist_ok=True)
    
    # Initialize components
    engine = EvolutionEngine(config)
    metrics_collector = MetricsCollector(
        storage_dir=config.paths.output_dir / "metrics",
        enable_telemetry=True
    )
    report_generator = ReportGenerator()
    
    # Run optimization with metrics tracking
    run_id = "metrics_demo"
    modules = ["MyProject.Performance"]
    
    try:
        # Start metrics tracking
        await metrics_collector.track_optimization_run(
            run_id, modules, {
                "population_size": config.optimization.population_size,
                "generations": config.optimization.generations,
                "time_budget": config.optimization.time_budget
            }
        )
        logger.info("📈 Metrics tracking started")
        
        # Run optimization
        result = await engine.run_evolution(
            modules=modules,
            source_path=Path.cwd(),
            time_budget=config.optimization.time_budget
        )
        
        # Complete metrics tracking
        await metrics_collector.complete_optimization_run(run_id, result)
        logger.info("📊 Metrics tracking completed")
        
        # Generate comprehensive reports
        logger.info("📝 Generating reports...")
        
        # HTML report with interactive charts
        html_report = await report_generator.generate_html_report(result)
        html_path = config.paths.output_dir / "detailed_report.html"
        with open(html_path, 'w') as f:
            f.write(html_report)
        logger.info(f"📄 HTML report: {html_path}")
        
        # Markdown summary
        md_report = report_generator.generate_markdown_summary(result)
        md_path = config.paths.output_dir / "summary_report.md"
        with open(md_path, 'w') as f:
            f.write(md_report)
        logger.info(f"📋 Markdown report: {md_path}")
        
        # Dashboard generation
        dashboard_html = await report_generator.generate_dashboard(
            results=[result],
            title="Simpulse Optimization Dashboard"
        )
        dashboard_path = config.paths.output_dir / "dashboard.html"
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)
        logger.info(f"📊 Dashboard: {dashboard_path}")
        
        # Export metrics in different formats
        metrics_json = await metrics_collector.export_metrics(run_id, format="json")
        json_path = config.paths.output_dir / "metrics.json"
        with open(json_path, 'w') as f:
            f.write(metrics_json)
        logger.info(f"📈 Metrics JSON: {json_path}")
        
        # Show summary
        logger.info("🎉 Metrics and reporting example completed!")
        logger.info(f"📈 Final improvement: {result.improvement_percent:.1f}%")
        logger.info(f"📊 Generated {4} report files")
        
    except Exception as e:
        logger.error(f"❌ Metrics and reporting example failed: {e}")


async def custom_configuration_example():
    """Example showing custom configuration and advanced settings."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("⚙️ Custom Configuration Example")
    
    # Advanced configuration with custom parameters
    config = Config(
        optimization=OptimizationConfig(
            population_size=40,  # Larger population for better diversity
            generations=60,      # More generations for better convergence
            time_budget=3600,    # 1 hour budget
            target_improvement=20.0,  # Ambitious target
            mutation_rate=0.25,   # Custom mutation rate
            crossover_rate=0.8,   # Higher crossover rate
            elite_size=5,         # Preserve top 5 candidates
            max_parallel_evaluations=6,  # More parallel workers
            
            # Custom fitness weights
            fitness_weights={
                "time": 0.5,      # 50% weight on execution time
                "memory": 0.2,    # 20% weight on memory usage
                "iterations": 0.2, # 20% weight on iteration count
                "depth": 0.1      # 10% weight on search depth
            }
        ),
        claude=ClaudeConfig(
            backend="claude_code",
            timeout_seconds=45,  # Longer timeout for complex queries
            max_retries=3,       # More retries for reliability
            temperature=0.7      # More creative mutations
        ),
        paths=PathConfig(
            output_dir=Path("./custom_output"),
            cache_dir=Path("./custom_cache"),
            log_dir=Path("./custom_logs")
        )
    )
    
    # Create directories
    for path in [config.paths.output_dir, config.paths.cache_dir, config.paths.log_dir]:
        path.mkdir(exist_ok=True)
    
    logger.info("⚙️ Using custom configuration:")
    logger.info(f"  Population: {config.optimization.population_size}")
    logger.info(f"  Generations: {config.optimization.generations}")
    logger.info(f"  Mutation rate: {config.optimization.mutation_rate}")
    logger.info(f"  Parallel workers: {config.optimization.max_parallel_evaluations}")
    
    engine = EvolutionEngine(config)
    
    # Target multiple modules with custom focus
    modules = [
        "MyProject.Core.Algebra",
        "MyProject.Core.Analysis", 
        "MyProject.Data.Structures",
        "MyProject.Algorithms.Graph"
    ]
    
    try:
        start_time = datetime.now()
        
        result = await engine.run_evolution(
            modules=modules,
            source_path=Path.cwd(),
            time_budget=config.optimization.time_budget
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("🎉 Custom optimization completed!")
        logger.info(f"📈 Improvement: {result.improvement_percent:.2f}%")
        logger.info(f"⏱️ Duration: {duration.total_seconds():.1f}s")
        logger.info(f"🧬 Generations: {result.total_generations}")
        logger.info(f"🔧 Total evaluations: {result.total_evaluations}")
        
        if result.best_candidate:
            logger.info(f"🏆 Best candidate mutations: {len(result.best_candidate.mutations)}")
            if result.best_candidate.fitness:
                logger.info(f"📊 Best fitness score: {result.best_candidate.fitness.composite_score:.4f}")
        
        # Performance analysis
        if result.improvement_percent > 15.0:
            logger.info("🚀 Excellent performance improvement achieved!")
        elif result.improvement_percent > 5.0:
            logger.info("✨ Good performance improvement achieved!")
        elif result.improvement_percent > 0:
            logger.info("📈 Minor improvement achieved!")
        else:
            logger.info("📊 No improvement found - rules already well-optimized")
            
    except Exception as e:
        logger.error(f"❌ Custom configuration example failed: {e}")


if __name__ == "__main__":
    print("🧬 Simpulse Advanced Features Examples")
    print("======================================")
    print()
    print("Choose an advanced example to run:")
    print("1. GitHub Integration (automated PRs)")
    print("2. Continuous Optimization (scheduled runs)")
    print("3. Metrics and Reporting (comprehensive analytics)")
    print("4. Custom Configuration (advanced settings)")
    print()
    
    choice = input("Enter choice (1-4): ").strip()
    
    examples = {
        "1": ("🐙 GitHub Integration", github_integration_example),
        "2": ("⚡ Continuous Optimization", continuous_optimization_example),
        "3": ("📊 Metrics and Reporting", metrics_and_reporting_example),
        "4": ("⚙️ Custom Configuration", custom_configuration_example)
    }
    
    if choice in examples:
        name, func = examples[choice]
        print(f"\n🚀 Running {name} example...")
        asyncio.run(func())
    else:
        print("❌ Invalid choice. Please run the script again and choose 1-4.")