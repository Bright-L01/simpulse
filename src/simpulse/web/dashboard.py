"""
Web-based monitoring dashboard for Simpulse optimization.

This module provides a FastAPI-based web dashboard for real-time monitoring,
historical trends, and interactive optimization management.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import uvicorn

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None
    FASTAPI_AVAILABLE = False

import plotly.graph_objects as go
import plotly.utils
from plotly.subplots import make_subplots

from ..deployment.continuous_optimizer import ContinuousOptimizer
from ..monitoring.metrics_collector import MetricsCollector
from ..analysis.impact_analyzer import ImpactAnalyzer
from ..config import Config

logger = logging.getLogger(__name__)


class OptimizationRequest(BaseModel):
    """Request model for triggering optimization."""
    modules: List[str]
    time_budget: int = 3600
    target_improvement: float = 15.0
    config_overrides: Dict[str, Any] = {}


class WebSocketManager:
    """Manager for WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Connect a new WebSocket client."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket client."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


class SimplulseDashboard:
    """Web-based monitoring dashboard for Simpulse."""
    
    def __init__(self, config: Config, port: int = 8080, host: str = "0.0.0.0"):
        """Initialize dashboard.
        
        Args:
            config: Simpulse configuration
            port: Port to run dashboard on
            host: Host to bind to
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for the web dashboard. Install with: pip install fastapi uvicorn")
        
        self.config = config
        self.port = port
        self.host = host
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Simpulse Dashboard",
            description="Real-time monitoring and management for Simpulse optimization",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize components
        self.continuous_optimizer = ContinuousOptimizer(config)
        self.metrics_collector = MetricsCollector(
            storage_dir=config.paths.output_dir / "metrics",
            enable_telemetry=True
        )
        self.impact_analyzer = ImpactAnalyzer(
            storage_dir=config.paths.output_dir / "impact"
        )
        
        # WebSocket manager
        self.websocket_manager = WebSocketManager()
        
        # Cache for dashboard data
        self.dashboard_cache: Dict[str, Any] = {}
        self.cache_timestamp: Optional[datetime] = None
        self.cache_ttl = timedelta(minutes=5)
        
        # Setup routes
        self.setup_routes()
        
        # Background tasks
        self.background_tasks = set()
    
    def setup_routes(self):
        """Setup API routes for dashboard."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Main dashboard page."""
            return self.render_dashboard_html()
        
        @self.app.get("/api/status")
        async def optimization_status():
            """Real-time optimization status."""
            try:
                status = await self._get_optimization_status()
                return JSONResponse(status)
            except Exception as e:
                logger.error(f"Failed to get optimization status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/trends")
        async def historical_trends():
            """Performance trends over time."""
            try:
                trends = await self._get_historical_trends()
                return JSONResponse(trends)
            except Exception as e:
                logger.error(f"Failed to get historical trends: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/metrics")
        async def current_metrics():
            """Current system metrics."""
            try:
                metrics = await self._get_current_metrics()
                return JSONResponse(metrics)
            except Exception as e:
                logger.error(f"Failed to get current metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/optimize")
        async def trigger_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
            """Trigger manual optimization."""
            try:
                run_id = await self.continuous_optimizer.trigger_manual_optimization(
                    modules=request.modules,
                    config_overrides=request.config_overrides
                )
                
                if run_id:
                    # Start monitoring task
                    background_tasks.add_task(self._monitor_optimization, run_id)
                    
                    return JSONResponse({
                        "success": True,
                        "run_id": run_id,
                        "message": "Optimization started successfully"
                    })
                else:
                    raise HTTPException(status_code=500, detail="Failed to start optimization")
                    
            except Exception as e:
                logger.error(f"Failed to trigger optimization: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/runs")
        async def list_runs():
            """List recent optimization runs."""
            try:
                runs = await self._get_recent_runs()
                return JSONResponse(runs)
            except Exception as e:
                logger.error(f"Failed to get recent runs: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/runs/{run_id}")
        async def get_run_details(run_id: str):
            """Get detailed information about a specific run."""
            try:
                details = self.continuous_optimizer.get_optimization_status(run_id)
                if details:
                    return JSONResponse(details)
                else:
                    raise HTTPException(status_code=404, detail="Run not found")
            except Exception as e:
                logger.error(f"Failed to get run details: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/runs/{run_id}/cancel")
        async def cancel_run(run_id: str):
            """Cancel a running optimization."""
            try:
                success = await self.continuous_optimizer.cancel_optimization(run_id)
                if success:
                    await self.websocket_manager.broadcast({
                        "type": "run_cancelled",
                        "run_id": run_id,
                        "timestamp": datetime.now().isoformat()
                    })
                    return JSONResponse({"success": True, "message": "Run cancelled successfully"})
                else:
                    raise HTTPException(status_code=404, detail="Run not found or cannot be cancelled")
            except Exception as e:
                logger.error(f"Failed to cancel run: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/triggers")
        async def list_triggers():
            """List optimization triggers."""
            try:
                triggers = self.continuous_optimizer.list_triggers()
                return JSONResponse(triggers)
            except Exception as e:
                logger.error(f"Failed to get triggers: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/impact/{run_id}")
        async def get_impact_analysis(run_id: str):
            """Get impact analysis for a specific run."""
            try:
                # This would load the impact analysis from storage
                # For now, return placeholder data
                impact = {
                    "run_id": run_id,
                    "cost_savings_usd": 25000,
                    "time_savings_hours": 120,
                    "energy_reduction_percent": 15.5,
                    "developer_productivity_gain": 12.3
                }
                return JSONResponse(impact)
            except Exception as e:
                logger.error(f"Failed to get impact analysis: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await self.websocket_manager.connect(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    # Handle WebSocket messages if needed
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
    
    async def _get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        # Check cache
        if (self.cache_timestamp and 
            datetime.now() - self.cache_timestamp < timedelta(seconds=30)):
            return self.dashboard_cache.get("status", {})
        
        # Get fresh data
        service_status = self.continuous_optimizer.get_service_status()
        
        # Get active runs with details
        active_runs = []
        for run_id in list(service_status.get("active_runs", {})):
            run_details = self.continuous_optimizer.get_optimization_status(run_id)
            if run_details:
                active_runs.append(run_details)
        
        status = {
            "service_running": service_status["is_running"],
            "active_runs": active_runs,
            "total_triggers": service_status["total_triggers"],
            "enabled_triggers": service_status["enabled_triggers"],
            "recent_runs": service_status["recent_runs"],
            "scheduler_running": service_status.get("scheduler_running", False),
            "timestamp": datetime.now().isoformat()
        }
        
        # Update cache
        self.dashboard_cache["status"] = status
        self.cache_timestamp = datetime.now()
        
        return status
    
    async def _get_historical_trends(self) -> Dict[str, Any]:
        """Get historical performance trends."""
        # This would load historical data from metrics collector
        # For now, generate sample trend data
        
        dates = [(datetime.now() - timedelta(days=30-i)).strftime('%Y-%m-%d') for i in range(30)]
        
        # Sample trend data
        improvements = [5 + (i % 7) * 2 + (i % 3) * 1.5 for i in range(30)]
        compilation_times = [60 - (i * 0.5) + (i % 5) * 2 for i in range(30)]
        
        # Create Plotly charts
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Performance Improvements Over Time', 'Compilation Times'),
            vertical_spacing=0.1
        )
        
        # Improvement trend
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=improvements,
                mode='lines+markers',
                name='Improvement %',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        
        # Compilation time trend
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=compilation_times,
                mode='lines+markers',
                name='Compilation Time (s)',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text="Simpulse Performance Trends (Last 30 Days)"
        )
        
        # Convert to JSON for frontend
        chart_json = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        
        return {
            "chart": chart_json,
            "summary": {
                "avg_improvement": sum(improvements) / len(improvements),
                "trend_direction": "improving" if improvements[-1] > improvements[0] else "declining",
                "best_day": dates[improvements.index(max(improvements))],
                "total_time_saved": sum(60 - t for t in compilation_times)
            }
        }
    
    async def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        import psutil
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Simpulse-specific metrics
        optimization_count = len(self.continuous_optimizer.run_history)
        
        return {
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            },
            "simpulse": {
                "total_optimizations": optimization_count,
                "active_runs": len(self.continuous_optimizer.active_runs),
                "triggers_count": len(self.continuous_optimizer.triggers),
                "cache_size_mb": len(str(self.dashboard_cache)) / (1024 * 1024)
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_recent_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent optimization runs."""
        runs = []
        
        # Get from active runs
        for run_id, run_info in self.continuous_optimizer.active_runs.items():
            runs.append({
                "run_id": run_id,
                "status": run_info.status.value,
                "trigger_id": run_info.trigger_id,
                "modules": run_info.modules,
                "started_at": run_info.started_at.isoformat(),
                "improvement_percent": run_info.improvement_percent,
                "is_active": True
            })
        
        # Get from history
        for run_info in self.continuous_optimizer.run_history[-limit:]:
            runs.append({
                "run_id": run_info.run_id,
                "status": run_info.status.value,
                "trigger_id": run_info.trigger_id,
                "modules": run_info.modules,
                "started_at": run_info.started_at.isoformat(),
                "completed_at": run_info.completed_at.isoformat() if run_info.completed_at else None,
                "improvement_percent": run_info.improvement_percent,
                "error_message": run_info.error_message,
                "pr_url": run_info.pr_url,
                "is_active": False
            })
        
        # Sort by start time (most recent first)
        runs.sort(key=lambda x: x["started_at"], reverse=True)
        
        return runs[:limit]
    
    async def _monitor_optimization(self, run_id: str):
        """Monitor optimization progress and broadcast updates."""
        logger.info(f"Starting monitoring for run {run_id}")
        
        while True:
            status = self.continuous_optimizer.get_optimization_status(run_id)
            
            if not status:
                logger.warning(f"No status found for run {run_id}")
                break
            
            # Broadcast status update
            await self.websocket_manager.broadcast({
                "type": "run_update",
                "run_id": run_id,
                "status": status,
                "timestamp": datetime.now().isoformat()
            })
            
            # Check if run is complete
            if status["status"] in ["completed", "failed", "cancelled"]:
                logger.info(f"Run {run_id} finished with status: {status['status']}")
                break
            
            # Wait before next check
            await asyncio.sleep(10)
    
    def render_dashboard_html(self) -> str:
        """Render the main dashboard HTML."""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simpulse Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        .status-running { color: #28a745; }
        .status-completed { color: #007bff; }
        .status-failed { color: #dc3545; }
        .status-pending { color: #ffc107; }
        .websocket-status {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
        }
        .websocket-connected { background: #d4edda; color: #155724; }
        .websocket-disconnected { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container-fluid">
            <span class="navbar-brand">🧬 Simpulse Dashboard</span>
            <div class="navbar-nav ms-auto">
                <span class="nav-link" id="last-update">Last update: --</span>
            </div>
        </div>
    </nav>

    <div id="websocket-status" class="websocket-status websocket-disconnected">
        ⚠️ Disconnected
    </div>

    <div class="container-fluid mt-3">
        <!-- Status Cards -->
        <div class="row">
            <div class="col-md-3">
                <div class="metric-card">
                    <h5>Service Status</h5>
                    <h3 id="service-status">--</h3>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h5>Active Runs</h5>
                    <h3 id="active-runs">--</h3>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h5>Total Triggers</h5>
                    <h3 id="total-triggers">--</h3>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h5>Recent Runs</h5>
                    <h3 id="recent-runs">--</h3>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="row mt-4">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h5>Performance Trends</h5>
                    </div>
                    <div class="card-body">
                        <div id="trends-chart" style="height: 400px;"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5>System Metrics</h5>
                    </div>
                    <div class="card-body">
                        <div id="system-metrics">
                            <p>CPU: <span id="cpu-usage">--%</span></p>
                            <p>Memory: <span id="memory-usage">--%</span></p>
                            <p>Disk: <span id="disk-usage">--%</span></p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Recent Runs Table -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5>Recent Optimization Runs</h5>
                        <button class="btn btn-primary btn-sm" onclick="showOptimizeModal()">
                            ➕ New Optimization
                        </button>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Run ID</th>
                                        <th>Status</th>
                                        <th>Improvement</th>
                                        <th>Modules</th>
                                        <th>Started</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody id="runs-table-body">
                                    <tr><td colspan="6" class="text-center">Loading...</td></tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Optimization Modal -->
    <div class="modal fade" id="optimizeModal" tabindex="-1">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Trigger Optimization</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <form id="optimize-form">
                        <div class="mb-3">
                            <label class="form-label">Modules (comma-separated)</label>
                            <input type="text" class="form-control" id="modules-input" 
                                   placeholder="Mathlib.Algebra.Group,Mathlib.Topology.Basic">
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Time Budget (seconds)</label>
                            <input type="number" class="form-control" id="time-budget-input" value="3600">
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Target Improvement (%)</label>
                            <input type="number" class="form-control" id="target-improvement-input" 
                                   value="15" step="0.1">
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-primary" onclick="triggerOptimization()">
                        Start Optimization
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // WebSocket connection
        let ws;
        let reconnectInterval;

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
                document.getElementById('websocket-status').textContent = '✅ Connected';
                document.getElementById('websocket-status').className = 'websocket-status websocket-connected';
                clearInterval(reconnectInterval);
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected');
                document.getElementById('websocket-status').textContent = '⚠️ Disconnected';
                document.getElementById('websocket-status').className = 'websocket-status websocket-disconnected';
                
                // Attempt to reconnect
                reconnectInterval = setInterval(connectWebSocket, 5000);
            };
        }

        function handleWebSocketMessage(data) {
            if (data.type === 'run_update') {
                updateRunInTable(data.run_id, data.status);
            } else if (data.type === 'run_cancelled') {
                showNotification(`Run ${data.run_id} cancelled`, 'warning');
                loadData();
            }
        }

        function updateRunInTable(runId, status) {
            const row = document.querySelector(`tr[data-run-id="${runId}"]`);
            if (row) {
                const statusCell = row.querySelector('.run-status');
                if (statusCell) {
                    statusCell.textContent = status.status;
                    statusCell.className = `run-status status-${status.status}`;
                }
                
                if (status.improvement_percent !== null) {
                    const improvementCell = row.querySelector('.run-improvement');
                    if (improvementCell) {
                        improvementCell.textContent = `${status.improvement_percent.toFixed(1)}%`;
                    }
                }
            }
        }

        // Data loading functions
        async function loadStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                document.getElementById('service-status').textContent = 
                    data.service_running ? '🟢 Running' : '🔴 Stopped';
                document.getElementById('active-runs').textContent = data.active_runs.length;
                document.getElementById('total-triggers').textContent = data.total_triggers;
                document.getElementById('recent-runs').textContent = data.recent_runs;
                
            } catch (error) {
                console.error('Failed to load status:', error);
            }
        }

        async function loadTrends() {
            try {
                const response = await fetch('/api/trends');
                const data = await response.json();
                
                if (data.chart) {
                    Plotly.newPlot('trends-chart', data.chart.data, data.chart.layout);
                }
            } catch (error) {
                console.error('Failed to load trends:', error);
            }
        }

        async function loadMetrics() {
            try {
                const response = await fetch('/api/metrics');
                const data = await response.json();
                
                document.getElementById('cpu-usage').textContent = `${data.system.cpu_percent.toFixed(1)}%`;
                document.getElementById('memory-usage').textContent = `${data.system.memory_percent.toFixed(1)}%`;
                document.getElementById('disk-usage').textContent = `${data.system.disk_percent.toFixed(1)}%`;
                
            } catch (error) {
                console.error('Failed to load metrics:', error);
            }
        }

        async function loadRuns() {
            try {
                const response = await fetch('/api/runs');
                const runs = await response.json();
                
                const tbody = document.getElementById('runs-table-body');
                tbody.innerHTML = '';
                
                runs.forEach(run => {
                    const row = document.createElement('tr');
                    row.setAttribute('data-run-id', run.run_id);
                    
                    const improvement = run.improvement_percent !== null ? 
                        `${run.improvement_percent.toFixed(1)}%` : '--';
                    
                    const startTime = new Date(run.started_at).toLocaleString();
                    
                    row.innerHTML = `
                        <td><code>${run.run_id}</code></td>
                        <td><span class="run-status status-${run.status}">${run.status}</span></td>
                        <td class="run-improvement">${improvement}</td>
                        <td>${run.modules.length} modules</td>
                        <td>${startTime}</td>
                        <td>
                            ${run.is_active ? 
                                `<button class="btn btn-sm btn-danger" onclick="cancelRun('${run.run_id}')">Cancel</button>` :
                                `<button class="btn btn-sm btn-info" onclick="viewRunDetails('${run.run_id}')">Details</button>`
                            }
                        </td>
                    `;
                    
                    tbody.appendChild(row);
                });
                
            } catch (error) {
                console.error('Failed to load runs:', error);
            }
        }

        // UI interaction functions
        function showOptimizeModal() {
            const modal = new bootstrap.Modal(document.getElementById('optimizeModal'));
            modal.show();
        }

        async function triggerOptimization() {
            const modules = document.getElementById('modules-input').value.split(',').map(m => m.trim());
            const timeBudget = parseInt(document.getElementById('time-budget-input').value);
            const targetImprovement = parseFloat(document.getElementById('target-improvement-input').value);
            
            try {
                const response = await fetch('/api/optimize', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        modules: modules,
                        time_budget: timeBudget,
                        target_improvement: targetImprovement
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showNotification(`Optimization started: ${result.run_id}`, 'success');
                    bootstrap.Modal.getInstance(document.getElementById('optimizeModal')).hide();
                    loadData();
                } else {
                    showNotification('Failed to start optimization', 'error');
                }
                
            } catch (error) {
                console.error('Failed to trigger optimization:', error);
                showNotification('Failed to start optimization', 'error');
            }
        }

        async function cancelRun(runId) {
            if (!confirm(`Cancel optimization run ${runId}?`)) return;
            
            try {
                const response = await fetch(`/api/runs/${runId}/cancel`, {method: 'POST'});
                const result = await response.json();
                
                if (result.success) {
                    showNotification('Run cancelled successfully', 'success');
                    loadData();
                } else {
                    showNotification('Failed to cancel run', 'error');
                }
            } catch (error) {
                console.error('Failed to cancel run:', error);
                showNotification('Failed to cancel run', 'error');
            }
        }

        function viewRunDetails(runId) {
            // Open details in new tab or modal
            window.open(`/api/runs/${runId}`, '_blank');
        }

        function showNotification(message, type) {
            // Simple notification system
            const alert = document.createElement('div');
            alert.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
            alert.style.position = 'fixed';
            alert.style.top = '60px';
            alert.style.right = '20px';
            alert.style.zIndex = '9999';
            alert.innerHTML = `
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            
            document.body.appendChild(alert);
            
            setTimeout(() => {
                if (alert.parentNode) {
                    alert.parentNode.removeChild(alert);
                }
            }, 5000);
        }

        function loadData() {
            loadStatus();
            loadTrends();
            loadMetrics();
            loadRuns();
            document.getElementById('last-update').textContent = `Last update: ${new Date().toLocaleTimeString()}`;
        }

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            connectWebSocket();
            loadData();
            
            // Auto-refresh every 30 seconds
            setInterval(loadData, 30000);
        });
    </script>
</body>
</html>
"""
        return html
    
    async def start_service(self):
        """Start the continuous optimizer service."""
        await self.continuous_optimizer.start_service()
        logger.info("Continuous optimizer service started")
    
    async def stop_service(self):
        """Stop the continuous optimizer service."""
        await self.continuous_optimizer.stop_service()
        logger.info("Continuous optimizer service stopped")
    
    def run(self):
        """Run the dashboard server."""
        logger.info(f"Starting Simpulse dashboard on http://{self.host}:{self.port}")
        
        # Configure uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        
        # Run the server
        try:
            server.run()
        except KeyboardInterrupt:
            logger.info("Dashboard stopped by user")
        except Exception as e:
            logger.error(f"Dashboard failed: {e}")
            raise


# Convenience function for running dashboard
async def run_dashboard(config: Config, port: int = 8080, host: str = "0.0.0.0"):
    """Run the Simpulse dashboard.
    
    Args:
        config: Simpulse configuration
        port: Port to run on
        host: Host to bind to
    """
    dashboard = SimplulseDashboard(config, port, host)
    
    # Start background services
    await dashboard.start_service()
    
    try:
        # Run the web server
        dashboard.run()
    finally:
        # Clean up
        await dashboard.stop_service()


if __name__ == "__main__":
    import sys
    from ..config import load_config
    
    # Simple CLI for testing
    config = load_config()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    
    asyncio.run(run_dashboard(config, port))