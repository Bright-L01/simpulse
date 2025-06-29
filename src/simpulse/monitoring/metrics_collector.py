"""Metrics collection and monitoring for Simpulse.

This module provides production monitoring, analytics collection,
and telemetry for optimization runs.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
import platform
import psutil
import threading

from ..evolution.models_v2 import EvolutionHistory, GenerationResult
from ..evolution.evolution_engine import OptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class ResourceUsage:
    """System resource usage metrics."""
    cpu_percent: float
    memory_mb: float
    disk_usage_mb: float
    peak_memory_mb: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    
@dataclass
class OptimizationMetrics:
    """Comprehensive metrics for an optimization run."""
    run_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Configuration
    modules: List[str] = field(default_factory=list)
    population_size: int = 0
    max_generations: int = 0
    
    # Results
    total_generations: int = 0
    total_evaluations: int = 0
    improvement_percent: float = 0.0
    best_fitness: float = 0.0
    success: bool = False
    
    # Performance
    execution_time: float = 0.0
    avg_generation_time: float = 0.0
    peak_memory_mb: float = 0.0
    total_cpu_time: float = 0.0
    
    # Evolution stats
    convergence_generation: Optional[int] = None
    final_diversity: float = 0.0
    mutations_applied: int = 0
    
    # Resource usage over time
    resource_timeline: List[ResourceUsage] = field(default_factory=list)
    
    # Environment info
    platform_info: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize platform info."""
        if not self.platform_info:
            self.platform_info = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": str(psutil.cpu_count()),
                "total_memory_gb": f"{psutil.virtual_memory().total / (1024**3):.1f}"
            }


@dataclass
class TelemetryEvent:
    """Anonymous telemetry event."""
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    session_id: str
    anonymous: bool = True


class MetricsCollector:
    """Collects and manages optimization metrics and telemetry."""
    
    def __init__(self, 
                 backend: str = "json",
                 storage_dir: Optional[Path] = None,
                 enable_telemetry: bool = False):
        """Initialize metrics collector.
        
        Args:
            backend: Storage backend (json, prometheus, influxdb)
            storage_dir: Directory for storing metrics
            enable_telemetry: Enable anonymous telemetry
        """
        self.backend = backend
        self.storage_dir = storage_dir or Path.home() / ".simpulse" / "metrics"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.enable_telemetry = enable_telemetry
        
        # Active tracking
        self.active_runs: Dict[str, OptimizationMetrics] = {}
        self.resource_monitors: Dict[str, threading.Thread] = {}
        self.telemetry_events: List[TelemetryEvent] = []
        
        # Session info
        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now()
        
        logger.info(f"Metrics collector initialized (backend: {backend}, session: {self.session_id})")
        
    async def track_optimization_run(self, 
                                   run_id: str,
                                   modules: List[str],
                                   config: Dict[str, Any]) -> OptimizationMetrics:
        """Start tracking an optimization run.
        
        Args:
            run_id: Unique run identifier
            modules: Modules being optimized
            config: Optimization configuration
            
        Returns:
            Metrics object for this run
        """
        metrics = OptimizationMetrics(
            run_id=run_id,
            start_time=datetime.now(),
            modules=modules,
            population_size=config.get('population_size', 0),
            max_generations=config.get('max_generations', 0)
        )
        
        self.active_runs[run_id] = metrics
        
        # Start resource monitoring
        self._start_resource_monitoring(run_id)
        
        # Record telemetry event
        if self.enable_telemetry:
            await self._record_telemetry_event("optimization_started", {
                "modules_count": len(modules),
                "population_size": metrics.population_size,
                "max_generations": metrics.max_generations
            })
            
        logger.info(f"Started tracking optimization run: {run_id}")
        return metrics
        
    async def update_generation_metrics(self, 
                                      run_id: str,
                                      generation: GenerationResult):
        """Update metrics with generation results.
        
        Args:
            run_id: Run identifier
            generation: Generation result data
        """
        if run_id not in self.active_runs:
            logger.warning(f"No active run found for ID: {run_id}")
            return
            
        metrics = self.active_runs[run_id]
        metrics.total_generations = generation.generation
        
        if generation.new_best_found:
            metrics.best_fitness = generation.best_fitness
            
        # Record telemetry for significant improvements
        if self.enable_telemetry and generation.new_best_found:
            await self._record_telemetry_event("new_best_found", {
                "generation": generation.generation,
                "fitness": generation.best_fitness,
                "improvement": generation.best_fitness - metrics.best_fitness
            })
            
    async def complete_optimization_run(self, 
                                      run_id: str,
                                      result: OptimizationResult):
        """Complete tracking for an optimization run.
        
        Args:
            run_id: Run identifier
            result: Final optimization result
        """
        if run_id not in self.active_runs:
            logger.warning(f"No active run found for ID: {run_id}")
            return
            
        metrics = self.active_runs[run_id]
        metrics.end_time = datetime.now()
        metrics.execution_time = result.execution_time
        metrics.total_evaluations = result.total_evaluations
        metrics.improvement_percent = result.improvement_percent
        metrics.success = result.success
        
        if result.best_candidate:
            metrics.mutations_applied = len(result.best_candidate.mutations)
            
        if result.history:
            if result.history.generations:
                avg_time = sum(g.evaluation_time for g in result.history.generations) / len(result.history.generations)
                metrics.avg_generation_time = avg_time
                metrics.final_diversity = result.history.generations[-1].diversity_score
                
            metrics.convergence_generation = result.history.convergence_generation
            
        # Stop resource monitoring
        self._stop_resource_monitoring(run_id)
        
        # Save metrics
        await self._save_metrics(metrics)
        
        # Record completion telemetry
        if self.enable_telemetry:
            await self._record_telemetry_event("optimization_completed", {
                "success": result.success,
                "improvement_percent": result.improvement_percent,
                "execution_time": result.execution_time,
                "total_evaluations": result.total_evaluations
            })
            
        # Move to completed runs
        completed_metrics = self.active_runs.pop(run_id)
        
        logger.info(f"Completed tracking optimization run: {run_id} "
                   f"(improvement: {result.improvement_percent:.1f}%)")
        
        return completed_metrics
        
    def _start_resource_monitoring(self, run_id: str):
        """Start background resource monitoring for a run.
        
        Args:
            run_id: Run identifier
        """
        def monitor_resources():
            while run_id in self.active_runs:
                try:
                    # Get current resource usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')
                    
                    usage = ResourceUsage(
                        cpu_percent=cpu_percent,
                        memory_mb=memory.used / (1024 * 1024),
                        disk_usage_mb=disk.used / (1024 * 1024),
                        peak_memory_mb=memory.used / (1024 * 1024)  # Simplified
                    )
                    
                    # Add to metrics
                    metrics = self.active_runs.get(run_id)
                    if metrics:
                        metrics.resource_timeline.append(usage)
                        metrics.peak_memory_mb = max(metrics.peak_memory_mb, usage.memory_mb)
                        
                except Exception as e:
                    logger.debug(f"Resource monitoring error: {e}")
                    
                time.sleep(5)  # Monitor every 5 seconds
                
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
        self.resource_monitors[run_id] = monitor_thread
        
    def _stop_resource_monitoring(self, run_id: str):
        """Stop resource monitoring for a run.
        
        Args:
            run_id: Run identifier
        """
        if run_id in self.resource_monitors:
            # Thread will stop automatically when run_id is removed from active_runs
            del self.resource_monitors[run_id]
            
    async def _save_metrics(self, metrics: OptimizationMetrics):
        """Save metrics to storage backend.
        
        Args:
            metrics: Metrics to save
        """
        try:
            if self.backend == "json":
                await self._save_json_metrics(metrics)
            elif self.backend == "prometheus":
                await self._save_prometheus_metrics(metrics)
            else:
                logger.warning(f"Unknown backend: {self.backend}")
                
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            
    async def _save_json_metrics(self, metrics: OptimizationMetrics):
        """Save metrics as JSON file.
        
        Args:
            metrics: Metrics to save
        """
        filename = f"optimization_{metrics.run_id}_{metrics.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        file_path = self.storage_dir / filename
        
        # Convert to serializable format
        data = asdict(metrics)
        
        # Convert datetime objects to ISO strings
        data['start_time'] = metrics.start_time.isoformat()
        if metrics.end_time:
            data['end_time'] = metrics.end_time.isoformat()
            
        # Convert resource timeline
        data['resource_timeline'] = [
            {
                **asdict(usage),
                'timestamp': usage.timestamp.isoformat()
            }
            for usage in metrics.resource_timeline
        ]
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.debug(f"Saved metrics to: {file_path}")
        
    async def _save_prometheus_metrics(self, metrics: OptimizationMetrics):
        """Save metrics in Prometheus format.
        
        Args:
            metrics: Metrics to save
        """
        # This would integrate with Prometheus pushgateway
        # For now, just log the metrics
        logger.info(f"Prometheus metrics for {metrics.run_id}:")
        logger.info(f"  simpulse_improvement_percent {metrics.improvement_percent}")
        logger.info(f"  simpulse_execution_time {metrics.execution_time}")
        logger.info(f"  simpulse_total_evaluations {metrics.total_evaluations}")
        logger.info(f"  simpulse_peak_memory_mb {metrics.peak_memory_mb}")
        
    async def _record_telemetry_event(self, event_type: str, data: Dict[str, Any]):
        """Record anonymous telemetry event.
        
        Args:
            event_type: Type of event
            data: Event data (anonymized)
        """
        if not self.enable_telemetry:
            return
            
        event = TelemetryEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            session_id=self.session_id,
            anonymous=True
        )
        
        self.telemetry_events.append(event)
        
        # Keep only recent events (memory management)
        if len(self.telemetry_events) > 1000:
            self.telemetry_events = self.telemetry_events[-500:]
            
    def export_metrics(self, 
                      format: str = "json",
                      time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """Export metrics for external systems.
        
        Args:
            format: Export format (json, csv, prometheus)
            time_range: Optional time range filter
            
        Returns:
            Exported metrics data
        """
        # Get metrics files
        metrics_files = list(self.storage_dir.glob("optimization_*.json"))
        
        if format == "json":
            return self._export_json_summary(metrics_files, time_range)
        elif format == "csv":
            return self._export_csv_summary(metrics_files, time_range)
        else:
            return json.dumps({"error": f"Unsupported format: {format}"})
            
    def _export_json_summary(self, 
                           metrics_files: List[Path],
                           time_range: Optional[Tuple[datetime, datetime]]) -> str:
        """Export JSON summary of metrics.
        
        Args:
            metrics_files: List of metrics files
            time_range: Optional time range filter
            
        Returns:
            JSON summary
        """
        summary = {
            "total_runs": len(metrics_files),
            "time_range": {
                "start": time_range[0].isoformat() if time_range else None,
                "end": time_range[1].isoformat() if time_range else None
            },
            "aggregated_metrics": {
                "total_improvements": 0.0,
                "avg_improvement": 0.0,
                "total_execution_time": 0.0,
                "success_rate": 0.0,
                "total_evaluations": 0
            },
            "runs": []
        }
        
        total_improvement = 0.0
        successful_runs = 0
        total_time = 0.0
        total_evaluations = 0
        
        for file_path in metrics_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Apply time filter if specified
                if time_range:
                    start_time = datetime.fromisoformat(data['start_time'])
                    if not (time_range[0] <= start_time <= time_range[1]):
                        continue
                        
                summary["runs"].append({
                    "run_id": data["run_id"],
                    "improvement_percent": data["improvement_percent"],
                    "execution_time": data["execution_time"],
                    "success": data["success"],
                    "modules": data["modules"]
                })
                
                # Aggregate metrics
                total_improvement += data["improvement_percent"]
                total_time += data["execution_time"]
                total_evaluations += data["total_evaluations"]
                
                if data["success"]:
                    successful_runs += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process metrics file {file_path}: {e}")
                
        # Calculate aggregates
        if summary["runs"]:
            summary["aggregated_metrics"]["avg_improvement"] = total_improvement / len(summary["runs"])
            summary["aggregated_metrics"]["total_improvements"] = total_improvement
            summary["aggregated_metrics"]["total_execution_time"] = total_time
            summary["aggregated_metrics"]["success_rate"] = successful_runs / len(summary["runs"])
            summary["aggregated_metrics"]["total_evaluations"] = total_evaluations
            
        return json.dumps(summary, indent=2)
        
    def _export_csv_summary(self, 
                          metrics_files: List[Path],
                          time_range: Optional[Tuple[datetime, datetime]]) -> str:
        """Export CSV summary of metrics.
        
        Args:
            metrics_files: List of metrics files
            time_range: Optional time range filter
            
        Returns:
            CSV data
        """
        csv_lines = [
            "run_id,start_time,improvement_percent,execution_time,total_evaluations,success,modules_count"
        ]
        
        for file_path in metrics_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Apply time filter if specified
                if time_range:
                    start_time = datetime.fromisoformat(data['start_time'])
                    if not (time_range[0] <= start_time <= time_range[1]):
                        continue
                        
                csv_lines.append(
                    f"{data['run_id']},"
                    f"{data['start_time']},"
                    f"{data['improvement_percent']},"
                    f"{data['execution_time']},"
                    f"{data['total_evaluations']},"
                    f"{data['success']},"
                    f"{len(data['modules'])}"
                )
                
            except Exception as e:
                logger.warning(f"Failed to process metrics file {file_path}: {e}")
                
        return "\n".join(csv_lines)
        
    async def send_telemetry(self, anonymous: bool = True) -> bool:
        """Send collected telemetry data.
        
        Args:
            anonymous: Whether to send anonymous data only
            
        Returns:
            True if telemetry sent successfully
        """
        if not self.enable_telemetry or not self.telemetry_events:
            return False
            
        try:
            # Filter events by anonymity setting
            events_to_send = [
                event for event in self.telemetry_events
                if not anonymous or event.anonymous
            ]
            
            if not events_to_send:
                return False
                
            # Prepare telemetry payload
            payload = {
                "session_id": self.session_id if not anonymous else "anonymous",
                "timestamp": datetime.now().isoformat(),
                "platform": platform.platform(),
                "events": [
                    {
                        "type": event.event_type,
                        "timestamp": event.timestamp.isoformat(),
                        "data": event.data
                    }
                    for event in events_to_send
                ]
            }
            
            # In a real implementation, this would send to a telemetry endpoint
            logger.info(f"Would send {len(events_to_send)} telemetry events")
            logger.debug(f"Telemetry payload: {json.dumps(payload, indent=2)}")
            
            # Clear sent events
            self.telemetry_events.clear()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send telemetry: {e}")
            return False
            
    def get_active_runs_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of currently active optimization runs.
        
        Returns:
            Dictionary of active run statuses
        """
        status = {}
        
        for run_id, metrics in self.active_runs.items():
            current_time = datetime.now()
            elapsed = (current_time - metrics.start_time).total_seconds()
            
            status[run_id] = {
                "start_time": metrics.start_time.isoformat(),
                "elapsed_seconds": elapsed,
                "modules": metrics.modules,
                "current_generation": metrics.total_generations,
                "max_generations": metrics.max_generations,
                "best_fitness": metrics.best_fitness,
                "peak_memory_mb": metrics.peak_memory_mb
            }
            
        return status
        
    def cleanup_old_metrics(self, max_age_days: int = 30):
        """Clean up old metrics files.
        
        Args:
            max_age_days: Maximum age of metrics files to keep
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        metrics_files = list(self.storage_dir.glob("optimization_*.json"))
        cleaned_count = 0
        
        for file_path in metrics_files:
            try:
                # Parse timestamp from filename
                timestamp_str = file_path.stem.split('_')[-2] + '_' + file_path.stem.split('_')[-1]
                file_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                if file_date < cutoff_date:
                    file_path.unlink()
                    cleaned_count += 1
                    
            except Exception as e:
                logger.debug(f"Failed to process file {file_path} for cleanup: {e}")
                
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old metrics files")