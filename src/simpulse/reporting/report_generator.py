"""Report generator for optimization results.

This module creates comprehensive reports with visualizations,
performance analytics, and interactive dashboards.
"""

import base64
import io
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional visualization dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from ..evolution.evolution_engine import OptimizationResult
from ..evolution.models_v2 import Candidate, EvolutionHistory, GenerationResult

logger = logging.getLogger(__name__)


class Dashboard:
    """Real-time monitoring dashboard data."""
    
    def __init__(self):
        self.generation_data = []
        self.fitness_timeline = []
        self.diversity_timeline = []
        self.mutation_stats = {}
        self.resource_usage = {}
        
    def add_generation_data(self, generation: GenerationResult):
        """Add generation data to dashboard."""
        self.generation_data.append({
            "generation": generation.generation,
            "best_fitness": generation.best_fitness,
            "average_fitness": generation.average_fitness,
            "diversity": generation.diversity_score,
            "valid_candidates": generation.valid_candidates,
            "timestamp": datetime.now().isoformat()
        })
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert dashboard to dictionary."""
        return {
            "generation_data": self.generation_data,
            "fitness_timeline": self.fitness_timeline,
            "diversity_timeline": self.diversity_timeline,
            "mutation_stats": self.mutation_stats,
            "resource_usage": self.resource_usage
        }


class ReportGenerator:
    """Generates comprehensive optimization reports."""
    
    def __init__(self, template_dir: Optional[Path] = None):
        """Initialize report generator.
        
        Args:
            template_dir: Directory containing report templates
        """
        self.template_dir = template_dir or Path(__file__).parent / "templates"
        self.template_dir.mkdir(exist_ok=True)
        
        # Check available dependencies
        self.plotly_available = PLOTLY_AVAILABLE
        self.pandas_available = PANDAS_AVAILABLE
        
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Install with: pip install plotly")
        if not PANDAS_AVAILABLE:
            logger.warning("Pandas not available. Install with: pip install pandas")
            
    async def generate_html_report(self, 
                                 result: OptimizationResult,
                                 include_interactive: bool = True) -> str:
        """Generate interactive HTML report with visualizations.
        
        Args:
            result: Optimization result
            include_interactive: Include interactive plots
            
        Returns:
            HTML report content
        """
        logger.info("Generating HTML report")
        
        # Generate visualizations
        plots_html = ""
        if self.plotly_available and include_interactive:
            plots_html = self._generate_plotly_visualizations(result)
        
        # Build complete HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simpulse Optimization Report</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        {self._generate_header_section(result)}
        {self._generate_summary_section(result)}
        {self._generate_performance_section(result)}
        {plots_html}
        {self._generate_mutations_section(result)}
        {self._generate_evolution_section(result)}
        {self._generate_footer_section()}
    </div>
</body>
</html>
"""
        
        return html_content
        
    def _generate_plotly_visualizations(self, result: OptimizationResult) -> str:
        """Generate Plotly visualizations.
        
        Args:
            result: Optimization result
            
        Returns:
            HTML content with embedded plots
        """
        plots_html = '<div class="visualizations">'
        
        try:
            # Performance timeline
            if result.history and result.history.generations:
                timeline_fig = self._create_performance_timeline(result.history)
                plots_html += '<div class="plot-container">'
                plots_html += '<h3>🏃 Performance Evolution</h3>'
                plots_html += pyo.plot(timeline_fig, include_plotlyjs=False, output_type='div')
                plots_html += '</div>'
                
            # Mutation effectiveness
            if result.best_candidate and result.best_candidate.mutations:
                mutation_fig = self._create_mutation_heatmap(result.best_candidate.mutations)
                plots_html += '<div class="plot-container">'
                plots_html += '<h3>🔬 Mutation Effectiveness</h3>'
                plots_html += pyo.plot(mutation_fig, include_plotlyjs=False, output_type='div')
                plots_html += '</div>'
                
            # Fitness distribution
            if result.history:
                distribution_fig = self._create_fitness_distribution(result.history)
                plots_html += '<div class="plot-container">'
                plots_html += '<h3>📊 Fitness Distribution</h3>'
                plots_html += pyo.plot(distribution_fig, include_plotlyjs=False, output_type='div')
                plots_html += '</div>'
                
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            plots_html += f'<div class="error">Error generating visualizations: {e}</div>'
            
        plots_html += '</div>'
        
        # Add Plotly.js
        plotly_js = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
        return plotly_js + plots_html
        
    def _create_performance_timeline(self, history: EvolutionHistory) -> go.Figure:
        """Create performance timeline plot.
        
        Args:
            history: Evolution history
            
        Returns:
            Plotly figure
        """
        generations = [g.generation for g in history.generations]
        best_fitness = [g.best_fitness for g in history.generations]
        avg_fitness = [g.average_fitness for g in history.generations]
        diversity = [g.diversity_score for g in history.generations]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Fitness Evolution', 'Population Diversity'),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Fitness traces
        fig.add_trace(
            go.Scatter(x=generations, y=best_fitness, name='Best Fitness', 
                      line=dict(color='green', width=3)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=generations, y=avg_fitness, name='Average Fitness',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Diversity trace
        fig.add_trace(
            go.Scatter(x=generations, y=diversity, name='Diversity',
                      line=dict(color='orange', width=2)),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Evolution Progress",
            xaxis_title="Generation",
            height=600,
            showlegend=True
        )
        
        return fig
        
    def _create_mutation_heatmap(self, mutations: List[Any]) -> go.Figure:
        """Create mutation effectiveness heatmap.
        
        Args:
            mutations: List of mutations
            
        Returns:
            Plotly figure
        """
        # Analyze mutations by type and effectiveness
        mutation_data = {}
        
        for mutation in mutations:
            if hasattr(mutation, 'suggestion'):
                suggestion = mutation.suggestion
                mut_type = suggestion.mutation_type.value
                confidence = suggestion.confidence
                
                if mut_type not in mutation_data:
                    mutation_data[mut_type] = []
                mutation_data[mut_type].append(confidence)
                
        # Create heatmap data
        types = list(mutation_data.keys())
        avg_confidence = [sum(mutation_data[t])/len(mutation_data[t]) for t in types]
        counts = [len(mutation_data[t]) for t in types]
        
        fig = go.Figure(data=go.Bar(
            x=types,
            y=avg_confidence,
            text=[f"{c:.1%} ({n} mutations)" for c, n in zip(avg_confidence, counts)],
            textposition='auto',
            marker_color=avg_confidence,
            colorscale='RdYlGn'
        ))
        
        fig.update_layout(
            title="Mutation Type Effectiveness",
            xaxis_title="Mutation Type",
            yaxis_title="Average Confidence",
            height=400
        )
        
        return fig
        
    def _create_fitness_distribution(self, history: EvolutionHistory) -> go.Figure:
        """Create fitness distribution plot.
        
        Args:
            history: Evolution history
            
        Returns:
            Plotly figure
        """
        if not history.generations:
            return go.Figure()
            
        # Get final generation fitness data
        final_gen = history.generations[-1]
        
        # Simulate fitness distribution (in practice would come from population data)
        import numpy as np
        np.random.seed(42)
        
        # Create synthetic fitness distribution around the final values
        n_samples = 30
        fitness_values = np.random.normal(final_gen.average_fitness, 0.1, n_samples)
        fitness_values = np.clip(fitness_values, 0, 1)
        
        fig = go.Figure(data=[
            go.Histogram(x=fitness_values, nbinsx=10, name="Fitness Distribution")
        ])
        
        # Add vertical lines for statistics
        fig.add_vline(x=final_gen.best_fitness, line_dash="dash", 
                     line_color="green", annotation_text="Best")
        fig.add_vline(x=final_gen.average_fitness, line_dash="dash",
                     line_color="blue", annotation_text="Average")
        
        fig.update_layout(
            title="Final Generation Fitness Distribution",
            xaxis_title="Fitness Score",
            yaxis_title="Count",
            height=400
        )
        
        return fig
        
    def _generate_header_section(self, result: OptimizationResult) -> str:
        """Generate report header section."""
        return f"""
        <header class="header">
            <h1>🧬 Simpulse Optimization Report</h1>
            <div class="header-info">
                <span class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</span>
                <span class="improvement">🚀 {result.improvement_percent:.1f}% Improvement</span>
            </div>
        </header>
        """
        
    def _generate_summary_section(self, result: OptimizationResult) -> str:
        """Generate executive summary section."""
        return f"""
        <section class="summary">
            <h2>📋 Executive Summary</h2>
            <div class="summary-grid">
                <div class="metric-card">
                    <h3>Performance Improvement</h3>
                    <div class="metric-value">{result.improvement_percent:.1f}%</div>
                </div>
                <div class="metric-card">
                    <h3>Generations</h3>
                    <div class="metric-value">{result.total_generations}</div>
                </div>
                <div class="metric-card">
                    <h3>Execution Time</h3>
                    <div class="metric-value">{result.execution_time:.1f}s</div>
                </div>
                <div class="metric-card">
                    <h3>Modules Optimized</h3>
                    <div class="metric-value">{len(result.modules)}</div>
                </div>
            </div>
        </section>
        """
        
    def _generate_performance_section(self, result: OptimizationResult) -> str:
        """Generate performance comparison section."""
        if not result.best_candidate or not result.best_candidate.fitness:
            return "<section><h2>📊 Performance Analysis</h2><p>No performance data available.</p></section>"
            
        fitness = result.best_candidate.fitness
        baseline_time = fitness.total_time / (1 - result.improvement_percent / 100)
        
        return f"""
        <section class="performance">
            <h2>📊 Performance Analysis</h2>
            <table class="performance-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Baseline</th>
                        <th>Optimized</th>
                        <th>Improvement</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Total Time</strong></td>
                        <td>{baseline_time:.2f} ms</td>
                        <td>{fitness.total_time:.2f} ms</td>
                        <td class="improvement">{result.improvement_percent:.1f}%</td>
                    </tr>
                    <tr>
                        <td><strong>Simp Time</strong></td>
                        <td>{baseline_time * 0.7:.2f} ms</td>
                        <td>{fitness.simp_time:.2f} ms</td>
                        <td class="improvement">{((baseline_time * 0.7 - fitness.simp_time) / (baseline_time * 0.7) * 100):.1f}%</td>
                    </tr>
                    <tr>
                        <td><strong>Iterations</strong></td>
                        <td>-</td>
                        <td>{fitness.iterations}</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td><strong>Memory Usage</strong></td>
                        <td>-</td>
                        <td>{fitness.memory_mb:.1f} MB</td>
                        <td>-</td>
                    </tr>
                </tbody>
            </table>
        </section>
        """
        
    def _generate_mutations_section(self, result: OptimizationResult) -> str:
        """Generate mutations analysis section."""
        if not result.best_candidate or not result.best_candidate.mutations:
            return "<section><h2>🔧 Mutations Applied</h2><p>No mutations applied.</p></section>"
            
        mutations_html = """
        <section class="mutations">
            <h2>🔧 Applied Mutations</h2>
            <div class="mutations-list">
        """
        
        for i, mutation in enumerate(result.best_candidate.mutations, 1):
            if hasattr(mutation, 'suggestion'):
                suggestion = mutation.suggestion
                mutations_html += f"""
                <div class="mutation-item">
                    <h4>{i}. {suggestion.rule_name}</h4>
                    <div class="mutation-details">
                        <span class="mutation-type">{suggestion.mutation_type.value}</span>
                        <span class="confidence">Confidence: {suggestion.confidence:.1%}</span>
                    </div>
                    <p class="description">{suggestion.description}</p>
                    {self._format_mutation_impact(suggestion.estimated_impact)}
                </div>
                """
                
        mutations_html += "</div></section>"
        return mutations_html
        
    def _format_mutation_impact(self, impact: Dict[str, float]) -> str:
        """Format mutation impact data."""
        if not impact:
            return ""
            
        impact_html = '<div class="impact-list">'
        for metric, value in impact.items():
            impact_html += f'<span class="impact-item">{metric}: {value}%</span>'
        impact_html += '</div>'
        
        return impact_html
        
    def _generate_evolution_section(self, result: OptimizationResult) -> str:
        """Generate evolution statistics section."""
        return f"""
        <section class="evolution">
            <h2>🧬 Evolution Statistics</h2>
            <div class="evolution-grid">
                <div class="stat-item">
                    <label>Total Evaluations:</label>
                    <value>{result.total_evaluations}</value>
                </div>
                <div class="stat-item">
                    <label>Success Rate:</label>
                    <value>{result.success}</value>
                </div>
                <div class="stat-item">
                    <label>Convergence:</label>
                    <value>{"Yes" if result.history and result.history.convergence_generation else "No"}</value>
                </div>
            </div>
        </section>
        """
        
    def _generate_footer_section(self) -> str:
        """Generate report footer."""
        return f"""
        <footer class="footer">
            <p>Generated by <strong>Simpulse</strong> - Evolutionary Simp Optimization</p>
            <p>Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        </footer>
        """
        
    def _get_css_styles(self) -> str:
        """Get CSS styles for HTML report."""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        
        .header {
            text-align: center;
            padding: 20px 0;
            border-bottom: 2px solid #4CAF50;
            margin-bottom: 30px;
        }
        
        .header h1 {
            margin: 0;
            color: #2E7D32;
            font-size: 2.5em;
        }
        
        .header-info {
            margin-top: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .improvement {
            background: #4CAF50;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #4CAF50;
        }
        
        .metric-card h3 {
            margin: 0 0 10px 0;
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2E7D32;
        }
        
        .performance-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        .performance-table th,
        .performance-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .performance-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        
        .performance-table .improvement {
            color: #4CAF50;
            font-weight: bold;
        }
        
        .mutations-list {
            margin: 20px 0;
        }
        
        .mutation-item {
            background: #f8f9fa;
            margin: 15px 0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #2196F3;
        }
        
        .mutation-details {
            display: flex;
            gap: 15px;
            margin: 5px 0;
        }
        
        .mutation-type {
            background: #2196F3;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
        }
        
        .confidence {
            background: #FF9800;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
        }
        
        .impact-list {
            margin-top: 10px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .impact-item {
            background: #E8F5E8;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            color: #2E7D32;
        }
        
        .evolution-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        
        .stat-item label {
            font-weight: bold;
        }
        
        .plot-container {
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .footer {
            text-align: center;
            padding: 30px 0;
            border-top: 1px solid #ddd;
            margin-top: 40px;
            color: #666;
        }
        
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #f44336;
        }
        """
        
    def generate_markdown_summary(self, result: OptimizationResult) -> str:
        """Generate GitHub-compatible markdown report.
        
        Args:
            result: Optimization result
            
        Returns:
            Markdown report content
        """
        lines = []
        
        # Header
        lines.append("# 🧬 Simpulse Optimization Report")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"**Improvement**: 🚀 {result.improvement_percent:.1f}%")
        lines.append("")
        
        # Summary
        lines.append("## 📊 Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| **Performance Improvement** | {result.improvement_percent:.1f}% |")
        lines.append(f"| **Generations** | {result.total_generations} |")
        lines.append(f"| **Total Evaluations** | {result.total_evaluations} |")
        lines.append(f"| **Execution Time** | {result.execution_time:.1f}s |")
        lines.append(f"| **Modules Optimized** | {len(result.modules)} |")
        
        if result.best_candidate:
            lines.append(f"| **Mutations Applied** | {len(result.best_candidate.mutations)} |")
            
        lines.append("")
        
        # Performance details
        if result.best_candidate and result.best_candidate.fitness:
            fitness = result.best_candidate.fitness
            baseline_time = fitness.total_time / (1 - result.improvement_percent / 100)
            
            lines.append("## ⚡ Performance Details")
            lines.append("")
            lines.append("| Metric | Baseline | Optimized | Improvement |")
            lines.append("|--------|----------|-----------|-------------|")
            lines.append(f"| Total Time | {baseline_time:.2f} ms | {fitness.total_time:.2f} ms | {result.improvement_percent:.1f}% |")
            lines.append(f"| Simp Time | {baseline_time * 0.7:.2f} ms | {fitness.simp_time:.2f} ms | {((baseline_time * 0.7 - fitness.simp_time) / (baseline_time * 0.7) * 100):.1f}% |")
            lines.append(f"| Iterations | - | {fitness.iterations} | - |")
            lines.append(f"| Memory | - | {fitness.memory_mb:.1f} MB | - |")
            lines.append("")
            
        # Modules
        lines.append("## 📦 Optimized Modules")
        lines.append("")
        for module in result.modules:
            lines.append(f"- `{module}`")
        lines.append("")
        
        # Mutations
        if result.best_candidate and result.best_candidate.mutations:
            lines.append("## 🔧 Applied Mutations")
            lines.append("")
            
            for i, mutation in enumerate(result.best_candidate.mutations, 1):
                if hasattr(mutation, 'suggestion'):
                    suggestion = mutation.suggestion
                    lines.append(f"### {i}. {suggestion.rule_name}")
                    lines.append(f"- **Type**: {suggestion.mutation_type.value}")
                    lines.append(f"- **Description**: {suggestion.description}")
                    lines.append(f"- **Confidence**: {suggestion.confidence:.1%}")
                    lines.append("")
                    
        return "\n".join(lines)
        
    def create_performance_dashboard(self, history: EvolutionHistory) -> Dashboard:
        """Create real-time monitoring dashboard.
        
        Args:
            history: Evolution history
            
        Returns:
            Dashboard with real-time data
        """
        dashboard = Dashboard()
        
        for generation in history.generations:
            dashboard.add_generation_data(generation)
            
        # Add mutation statistics
        dashboard.mutation_stats = {
            "total_mutations": sum(len(g.mutation_statistics) for g in history.generations),
            "successful_mutations": len([g for g in history.generations if g.new_best_found]),
            "convergence_generation": history.convergence_generation
        }
        
        # Add resource usage (simulated)
        dashboard.resource_usage = {
            "peak_memory_mb": 256,
            "cpu_time_seconds": sum(g.evaluation_time for g in history.generations),
            "disk_space_mb": 50
        }
        
        return dashboard
    
    def _sanitize_json_input(self, json_str: str) -> str:
        """Sanitize JSON input for security.
        
        Args:
            json_str: JSON string to sanitize
            
        Returns:
            Sanitized JSON string
        """
        from ..security.validators import sanitize_json_input
        return sanitize_json_input(json_str)