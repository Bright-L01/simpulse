"""Continuous optimization service for automated simp rule optimization.

This module provides a service for continuous optimization that can be
triggered by commits, scheduled runs, or manual triggers.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    SCHEDULER_AVAILABLE = True
except ImportError:
    AsyncIOScheduler = None
    CronTrigger = None
    IntervalTrigger = None
    SCHEDULER_AVAILABLE = False

from ..config import Config
from ..evolution.evolution_engine import EvolutionEngine
from ..monitoring.metrics_collector import MetricsCollector
from ..deployment.github_action import GitHubActionRunner

logger = logging.getLogger(__name__)


class RunStatus(Enum):
    """Status of optimization runs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TriggerType(Enum):
    """Types of optimization triggers."""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    COMMIT_HOOK = "commit_hook"
    PERFORMANCE_THRESHOLD = "performance_threshold"


@dataclass
class OptimizationTrigger:
    """Configuration for optimization triggers."""
    trigger_id: str
    trigger_type: TriggerType
    modules: List[str]
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    # Scheduling
    cron_expression: Optional[str] = None
    interval_minutes: Optional[int] = None
    
    # Commit hook
    watched_paths: List[str] = field(default_factory=list)
    branch_patterns: List[str] = field(default_factory=lambda: ["main", "master"])
    
    # Performance threshold
    performance_threshold: Optional[float] = None
    baseline_window_hours: int = 24
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    total_runs: int = 0


@dataclass
class RunInfo:
    """Information about an optimization run."""
    run_id: str
    trigger_id: str
    status: RunStatus
    modules: List[str]
    started_at: datetime
    completed_at: Optional[datetime] = None
    improvement_percent: Optional[float] = None
    error_message: Optional[str] = None
    pr_url: Optional[str] = None
    result_path: Optional[Path] = None


class ContinuousOptimizer:
    """Service for continuous simp rule optimization."""
    
    def __init__(self, config: Config):
        """Initialize continuous optimizer.
        
        Args:
            config: Simpulse configuration
        """
        self.config = config
        self.triggers: Dict[str, OptimizationTrigger] = {}
        self.active_runs: Dict[str, RunInfo] = {}
        self.run_history: List[RunInfo] = []
        
        # Initialize scheduler if available
        if SCHEDULER_AVAILABLE:
            self.scheduler = AsyncIOScheduler()
        else:
            self.scheduler = None
            logger.warning("APScheduler not available. Install with: pip install apscheduler")
        
        # Initialize components
        self.evolution_engine = EvolutionEngine(config)
        self.metrics_collector = MetricsCollector(
            storage_dir=config.paths.output_dir / "metrics",
            enable_telemetry=True
        )
        self.github_runner = GitHubActionRunner()
        
        # Service state
        self.is_running = False
        self.max_concurrent_runs = config.optimization.max_parallel_evaluations
        
    async def start_service(self):
        """Start the continuous optimization service."""
        if self.is_running:
            logger.warning("Service is already running")
            return
            
        logger.info("Starting continuous optimization service")
        
        if self.scheduler:
            self.scheduler.start()
            logger.info("✓ Scheduler started")
        
        # Load existing triggers
        await self._load_triggers()
        
        self.is_running = True
        logger.info("✓ Continuous optimization service started")
        
    async def stop_service(self):
        """Stop the continuous optimization service."""
        if not self.is_running:
            return
            
        logger.info("Stopping continuous optimization service")
        
        # Cancel active runs
        for run_id in list(self.active_runs.keys()):
            await self.cancel_optimization(run_id)
        
        # Stop scheduler
        if self.scheduler:
            self.scheduler.shutdown()
            logger.info("✓ Scheduler stopped")
        
        # Save triggers
        await self._save_triggers()
        
        self.is_running = False
        logger.info("✓ Continuous optimization service stopped")
        
    async def schedule_optimization(self, 
                                  trigger_id: str,
                                  modules: List[str],
                                  cron_expression: str,
                                  config_overrides: Optional[Dict[str, Any]] = None) -> bool:
        """Schedule regular optimization runs.
        
        Args:
            trigger_id: Unique trigger identifier
            modules: Modules to optimize
            cron_expression: Cron expression for scheduling
            config_overrides: Configuration overrides
            
        Returns:
            True if scheduled successfully
        """
        if not self.scheduler:
            logger.error("Scheduler not available")
            return False
            
        trigger = OptimizationTrigger(
            trigger_id=trigger_id,
            trigger_type=TriggerType.SCHEDULED,
            modules=modules,
            config=config_overrides or {},
            cron_expression=cron_expression
        )
        
        try:
            # Add to scheduler
            self.scheduler.add_job(
                self._trigger_optimization,
                trigger=CronTrigger.from_crontab(cron_expression),
                args=[trigger_id],
                id=trigger_id,
                replace_existing=True
            )
            
            self.triggers[trigger_id] = trigger
            await self._save_triggers()
            
            logger.info(f"✓ Scheduled optimization '{trigger_id}' with cron: {cron_expression}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule optimization: {e}")
            return False
            
    async def schedule_interval_optimization(self,
                                           trigger_id: str,
                                           modules: List[str],
                                           interval_minutes: int,
                                           config_overrides: Optional[Dict[str, Any]] = None) -> bool:
        """Schedule optimization at regular intervals.
        
        Args:
            trigger_id: Unique trigger identifier
            modules: Modules to optimize
            interval_minutes: Interval in minutes
            config_overrides: Configuration overrides
            
        Returns:
            True if scheduled successfully
        """
        if not self.scheduler:
            logger.error("Scheduler not available")
            return False
            
        trigger = OptimizationTrigger(
            trigger_id=trigger_id,
            trigger_type=TriggerType.SCHEDULED,
            modules=modules,
            config=config_overrides or {},
            interval_minutes=interval_minutes
        )
        
        try:
            # Add to scheduler
            self.scheduler.add_job(
                self._trigger_optimization,
                trigger=IntervalTrigger(minutes=interval_minutes),
                args=[trigger_id],
                id=trigger_id,
                replace_existing=True
            )
            
            self.triggers[trigger_id] = trigger
            await self._save_triggers()
            
            logger.info(f"✓ Scheduled optimization '{trigger_id}' every {interval_minutes} minutes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule interval optimization: {e}")
            return False
            
    async def setup_commit_hook(self,
                              trigger_id: str,
                              modules: List[str],
                              watched_paths: List[str],
                              branch_patterns: Optional[List[str]] = None) -> bool:
        """Setup optimization trigger for commit hooks.
        
        Args:
            trigger_id: Unique trigger identifier
            modules: Modules to optimize
            watched_paths: Paths to watch for changes
            branch_patterns: Branch patterns to watch
            
        Returns:
            True if setup successfully
        """
        trigger = OptimizationTrigger(
            trigger_id=trigger_id,
            trigger_type=TriggerType.COMMIT_HOOK,
            modules=modules,
            watched_paths=watched_paths,
            branch_patterns=branch_patterns or ["main", "master"]
        )
        
        self.triggers[trigger_id] = trigger
        await self._save_triggers()
        
        logger.info(f"✓ Setup commit hook '{trigger_id}' for paths: {watched_paths}")
        return True
        
    async def handle_commit_hook(self, 
                               commit_sha: str,
                               branch: str,
                               changed_files: List[str]) -> List[str]:
        """Handle commit hook and trigger relevant optimizations.
        
        Args:
            commit_sha: Commit SHA
            branch: Branch name
            changed_files: List of changed files
            
        Returns:
            List of triggered run IDs
        """
        triggered_runs = []
        
        for trigger_id, trigger in self.triggers.items():
            if trigger.trigger_type != TriggerType.COMMIT_HOOK or not trigger.enabled:
                continue
                
            # Check branch patterns
            if not any(branch.match(pattern) for pattern in trigger.branch_patterns):
                continue
                
            # Check if any watched paths were changed
            relevant_changes = False
            for changed_file in changed_files:
                if any(changed_file.startswith(path) for path in trigger.watched_paths):
                    relevant_changes = True
                    break
                    
            if relevant_changes:
                run_id = await self._trigger_optimization(trigger_id, {
                    "commit_sha": commit_sha,
                    "branch": branch,
                    "changed_files": changed_files
                })
                
                if run_id:
                    triggered_runs.append(run_id)
                    
        logger.info(f"Commit {commit_sha[:8]} triggered {len(triggered_runs)} optimization runs")
        return triggered_runs
        
    async def trigger_manual_optimization(self,
                                        modules: List[str],
                                        config_overrides: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Trigger a manual optimization run.
        
        Args:
            modules: Modules to optimize
            config_overrides: Configuration overrides
            
        Returns:
            Run ID if triggered successfully
        """
        trigger_id = f"manual_{uuid.uuid4().hex[:8]}"
        
        trigger = OptimizationTrigger(
            trigger_id=trigger_id,
            trigger_type=TriggerType.MANUAL,
            modules=modules,
            config=config_overrides or {}
        )
        
        self.triggers[trigger_id] = trigger
        
        return await self._trigger_optimization(trigger_id)
        
    async def _trigger_optimization(self, 
                                  trigger_id: str,
                                  context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Internal method to trigger optimization.
        
        Args:
            trigger_id: Trigger identifier
            context: Additional context for the run
            
        Returns:
            Run ID if triggered successfully
        """
        if trigger_id not in self.triggers:
            logger.error(f"Unknown trigger: {trigger_id}")
            return None
            
        trigger = self.triggers[trigger_id]
        
        # Check if we're at max concurrent runs
        if len(self.active_runs) >= self.max_concurrent_runs:
            logger.warning(f"Max concurrent runs reached, skipping trigger {trigger_id}")
            return None
            
        # Create run info
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        run_info = RunInfo(
            run_id=run_id,
            trigger_id=trigger_id,
            status=RunStatus.PENDING,
            modules=trigger.modules,
            started_at=datetime.now()
        )
        
        self.active_runs[run_id] = run_info
        
        # Update trigger statistics
        trigger.last_triggered = datetime.now()
        trigger.total_runs += 1
        
        # Start optimization asynchronously
        asyncio.create_task(self._run_optimization(run_id, trigger, context))
        
        logger.info(f"Triggered optimization run {run_id} for trigger {trigger_id}")
        return run_id
        
    async def _run_optimization(self, 
                              run_id: str,
                              trigger: OptimizationTrigger,
                              context: Optional[Dict[str, Any]] = None):
        """Run the optimization process.
        
        Args:
            run_id: Run identifier
            trigger: Trigger configuration
            context: Additional context
        """
        run_info = self.active_runs[run_id]
        run_info.status = RunStatus.RUNNING
        
        try:
            logger.info(f"Starting optimization run {run_id}")
            
            # Merge configuration
            optimization_config = self.config.optimization.__dict__.copy()
            optimization_config.update(trigger.config)
            
            # Start metrics tracking
            await self.metrics_collector.track_optimization_run(
                run_id, trigger.modules, optimization_config
            )
            
            # Run evolution
            result = await self.evolution_engine.run_evolution(
                modules=trigger.modules,
                source_path=Path.cwd(),
                time_budget=optimization_config.get('time_budget', 3600)
            )
            
            # Complete metrics tracking
            await self.metrics_collector.complete_optimization_run(run_id, result)
            
            # Update run info
            run_info.status = RunStatus.COMPLETED
            run_info.completed_at = datetime.now()
            run_info.improvement_percent = result.improvement_percent
            
            # Create PR if successful and configured
            if result.success and optimization_config.get('create_pr', False):
                pr_url = await self.github_runner.create_optimization_pr(result)
                run_info.pr_url = pr_url
                
            # Save results
            results_dir = self.config.paths.output_dir / "continuous_runs" / run_id
            results_dir.mkdir(parents=True, exist_ok=True)
            
            results_path = results_dir / "results.json"
            with open(results_path, 'w') as f:
                json.dump({
                    "run_id": run_id,
                    "trigger_id": trigger.trigger_id,
                    "improvement_percent": result.improvement_percent,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "modules": result.modules,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
                
            run_info.result_path = results_path
            
            logger.info(f"✓ Optimization run {run_id} completed: {result.improvement_percent:.1f}% improvement")
            
        except Exception as e:
            logger.error(f"Optimization run {run_id} failed: {e}")
            run_info.status = RunStatus.FAILED
            run_info.error_message = str(e)
            run_info.completed_at = datetime.now()
            
        finally:
            # Move to history
            self.run_history.append(run_info)
            del self.active_runs[run_id]
            
            # Cleanup old history (keep last 100 runs)
            if len(self.run_history) > 100:
                self.run_history = self.run_history[-100:]
                
    async def cancel_optimization(self, run_id: str) -> bool:
        """Cancel a running optimization.
        
        Args:
            run_id: Run identifier
            
        Returns:
            True if cancelled successfully
        """
        if run_id not in self.active_runs:
            return False
            
        run_info = self.active_runs[run_id]
        run_info.status = RunStatus.CANCELLED
        run_info.completed_at = datetime.now()
        
        # Move to history
        self.run_history.append(run_info)
        del self.active_runs[run_id]
        
        logger.info(f"Cancelled optimization run {run_id}")
        return True
        
    def get_optimization_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an optimization run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Status information or None if not found
        """
        # Check active runs
        if run_id in self.active_runs:
            run_info = self.active_runs[run_id]
            return {
                "run_id": run_id,
                "status": run_info.status.value,
                "trigger_id": run_info.trigger_id,
                "modules": run_info.modules,
                "started_at": run_info.started_at.isoformat(),
                "elapsed_seconds": (datetime.now() - run_info.started_at).total_seconds()
            }
            
        # Check history
        for run_info in self.run_history:
            if run_info.run_id == run_id:
                return {
                    "run_id": run_id,
                    "status": run_info.status.value,
                    "trigger_id": run_info.trigger_id,
                    "modules": run_info.modules,
                    "started_at": run_info.started_at.isoformat(),
                    "completed_at": run_info.completed_at.isoformat() if run_info.completed_at else None,
                    "improvement_percent": run_info.improvement_percent,
                    "error_message": run_info.error_message,
                    "pr_url": run_info.pr_url
                }
                
        return None
        
    def get_service_status(self) -> Dict[str, Any]:
        """Get overall service status.
        
        Returns:
            Service status information
        """
        return {
            "is_running": self.is_running,
            "active_runs": len(self.active_runs),
            "total_triggers": len(self.triggers),
            "enabled_triggers": len([t for t in self.triggers.values() if t.enabled]),
            "recent_runs": len([r for r in self.run_history if r.started_at > datetime.now() - timedelta(hours=24)]),
            "scheduler_running": self.scheduler.running if self.scheduler else False
        }
        
    def list_triggers(self) -> List[Dict[str, Any]]:
        """List all optimization triggers.
        
        Returns:
            List of trigger information
        """
        return [
            {
                "trigger_id": trigger.trigger_id,
                "trigger_type": trigger.trigger_type.value,
                "modules": trigger.modules,
                "enabled": trigger.enabled,
                "cron_expression": trigger.cron_expression,
                "interval_minutes": trigger.interval_minutes,
                "total_runs": trigger.total_runs,
                "last_triggered": trigger.last_triggered.isoformat() if trigger.last_triggered else None
            }
            for trigger in self.triggers.values()
        ]
        
    async def enable_trigger(self, trigger_id: str) -> bool:
        """Enable a trigger.
        
        Args:
            trigger_id: Trigger identifier
            
        Returns:
            True if enabled successfully
        """
        if trigger_id not in self.triggers:
            return False
            
        self.triggers[trigger_id].enabled = True
        await self._save_triggers()
        
        logger.info(f"Enabled trigger {trigger_id}")
        return True
        
    async def disable_trigger(self, trigger_id: str) -> bool:
        """Disable a trigger.
        
        Args:
            trigger_id: Trigger identifier
            
        Returns:
            True if disabled successfully
        """
        if trigger_id not in self.triggers:
            return False
            
        self.triggers[trigger_id].enabled = False
        
        # Remove from scheduler if it's a scheduled trigger
        if self.scheduler and trigger_id in [job.id for job in self.scheduler.get_jobs()]:
            self.scheduler.remove_job(trigger_id)
            
        await self._save_triggers()
        
        logger.info(f"Disabled trigger {trigger_id}")
        return True
        
    async def remove_trigger(self, trigger_id: str) -> bool:
        """Remove a trigger completely.
        
        Args:
            trigger_id: Trigger identifier
            
        Returns:
            True if removed successfully
        """
        if trigger_id not in self.triggers:
            return False
            
        # Remove from scheduler
        if self.scheduler and trigger_id in [job.id for job in self.scheduler.get_jobs()]:
            self.scheduler.remove_job(trigger_id)
            
        # Remove from triggers
        del self.triggers[trigger_id]
        await self._save_triggers()
        
        logger.info(f"Removed trigger {trigger_id}")
        return True
        
    async def _save_triggers(self):
        """Save triggers to persistent storage."""
        triggers_file = self.config.paths.output_dir / "continuous_triggers.json"
        
        triggers_data = []
        for trigger in self.triggers.values():
            data = {
                "trigger_id": trigger.trigger_id,
                "trigger_type": trigger.trigger_type.value,
                "modules": trigger.modules,
                "config": trigger.config,
                "enabled": trigger.enabled,
                "cron_expression": trigger.cron_expression,
                "interval_minutes": trigger.interval_minutes,
                "watched_paths": trigger.watched_paths,
                "branch_patterns": trigger.branch_patterns,
                "performance_threshold": trigger.performance_threshold,
                "baseline_window_hours": trigger.baseline_window_hours,
                "created_at": trigger.created_at.isoformat(),
                "last_triggered": trigger.last_triggered.isoformat() if trigger.last_triggered else None,
                "total_runs": trigger.total_runs
            }
            triggers_data.append(data)
            
        with open(triggers_file, 'w') as f:
            json.dump(triggers_data, f, indent=2)
            
    async def _load_triggers(self):
        """Load triggers from persistent storage."""
        triggers_file = self.config.paths.output_dir / "continuous_triggers.json"
        
        if not triggers_file.exists():
            return
            
        try:
            with open(triggers_file, 'r') as f:
                triggers_data = json.load(f)
                
            for data in triggers_data:
                trigger = OptimizationTrigger(
                    trigger_id=data["trigger_id"],
                    trigger_type=TriggerType(data["trigger_type"]),
                    modules=data["modules"],
                    config=data["config"],
                    enabled=data["enabled"],
                    cron_expression=data.get("cron_expression"),
                    interval_minutes=data.get("interval_minutes"),
                    watched_paths=data.get("watched_paths", []),
                    branch_patterns=data.get("branch_patterns", ["main", "master"]),
                    performance_threshold=data.get("performance_threshold"),
                    baseline_window_hours=data.get("baseline_window_hours", 24),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    last_triggered=datetime.fromisoformat(data["last_triggered"]) if data.get("last_triggered") else None,
                    total_runs=data.get("total_runs", 0)
                )
                
                self.triggers[trigger.trigger_id] = trigger
                
                # Re-add to scheduler if enabled and scheduled
                if (trigger.enabled and 
                    trigger.trigger_type == TriggerType.SCHEDULED and 
                    self.scheduler):
                    
                    if trigger.cron_expression:
                        self.scheduler.add_job(
                            self._trigger_optimization,
                            trigger=CronTrigger.from_crontab(trigger.cron_expression),
                            args=[trigger.trigger_id],
                            id=trigger.trigger_id,
                            replace_existing=True
                        )
                    elif trigger.interval_minutes:
                        self.scheduler.add_job(
                            self._trigger_optimization,
                            trigger=IntervalTrigger(minutes=trigger.interval_minutes),
                            args=[trigger.trigger_id],
                            id=trigger.trigger_id,
                            replace_existing=True
                        )
                        
            logger.info(f"Loaded {len(self.triggers)} triggers from storage")
            
        except Exception as e:
            logger.error(f"Failed to load triggers: {e}")