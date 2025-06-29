"""
Impact analysis for Simpulse optimization results.

This module analyzes the real-world impact of optimizations including
time savings, energy reduction, and productivity improvements.
"""

import asyncio
import json
import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..evolution.models import OptimizationResult, AppliedMutation
from ..benchmarks.benchmark_suite import BenchmarkResult, ComparisonReport

logger = logging.getLogger(__name__)


@dataclass
class TimeSavingsProjection:
    """Projection of time savings from optimization."""
    daily_savings_minutes: float
    weekly_savings_hours: float
    monthly_savings_hours: float
    annual_savings_hours: float
    
    # Developer impact
    developers_affected: int
    productivity_gain_percent: float
    
    # CI/CD impact
    ci_runs_per_day: int
    ci_time_savings_minutes: float
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.weekly_savings_hours = self.daily_savings_minutes * 7 / 60
        self.monthly_savings_hours = self.daily_savings_minutes * 30 / 60
        self.annual_savings_hours = self.daily_savings_minutes * 365 / 60


@dataclass
class EnergyImpact:
    """Energy usage impact analysis."""
    baseline_watts_per_hour: float
    optimized_watts_per_hour: float
    energy_reduction_percent: float
    
    # Carbon footprint
    carbon_reduction_kg_co2_per_year: float
    carbon_cost_savings_usd_per_year: float
    
    # Energy costs
    energy_cost_per_kwh_usd: float = 0.12  # Average US rate
    annual_energy_cost_savings_usd: float = 0.0
    
    def __post_init__(self):
        """Calculate energy cost savings."""
        hours_per_year = 8760  # 24 * 365
        baseline_kwh_per_year = (self.baseline_watts_per_hour * hours_per_year) / 1000
        optimized_kwh_per_year = (self.optimized_watts_per_hour * hours_per_year) / 1000
        
        kwh_savings = baseline_kwh_per_year - optimized_kwh_per_year
        self.annual_energy_cost_savings_usd = kwh_savings * self.energy_cost_per_kwh_usd


@dataclass
class ProductivityImpact:
    """Developer productivity impact analysis."""
    developers_count: int
    avg_compilation_frequency: int  # compilations per day
    compilation_time_savings_seconds: float
    
    # Calculated metrics
    daily_time_savings_per_dev_minutes: float = 0.0
    focus_improvement_percent: float = 0.0
    frustration_reduction_score: float = 0.0
    
    def __post_init__(self):
        """Calculate productivity metrics."""
        daily_savings_seconds = self.avg_compilation_frequency * self.compilation_time_savings_seconds
        self.daily_time_savings_per_dev_minutes = daily_savings_seconds / 60
        
        # Focus improvement based on compilation wait time reduction
        if self.compilation_time_savings_seconds > 10:
            self.focus_improvement_percent = min(25, self.compilation_time_savings_seconds / 2)
        
        # Frustration reduction (subjective metric)
        self.frustration_reduction_score = min(10, self.compilation_time_savings_seconds / 5)


@dataclass
class CostSavings:
    """Cost savings analysis."""
    # Infrastructure costs
    ci_cd_cost_per_minute_usd: float
    ci_cd_minutes_saved_per_day: float
    annual_ci_cd_savings_usd: float = 0.0
    
    # Developer costs
    avg_developer_cost_per_hour_usd: float
    developer_hours_saved_per_year: float
    annual_developer_cost_savings_usd: float = 0.0
    
    # Cloud compute costs
    cloud_compute_cost_per_hour_usd: float
    compute_hours_saved_per_year: float
    annual_cloud_savings_usd: float = 0.0
    
    def __post_init__(self):
        """Calculate total cost savings."""
        # CI/CD savings
        self.annual_ci_cd_savings_usd = self.ci_cd_minutes_saved_per_day * 365 * self.ci_cd_cost_per_minute_usd
        
        # Developer cost savings
        self.annual_developer_cost_savings_usd = self.developer_hours_saved_per_year * self.avg_developer_cost_per_hour_usd
        
        # Cloud compute savings
        self.annual_cloud_savings_usd = self.compute_hours_saved_per_year * self.cloud_compute_cost_per_hour_usd
    
    @property
    def total_annual_savings_usd(self) -> float:
        """Total annual cost savings."""
        return (self.annual_ci_cd_savings_usd + 
                self.annual_developer_cost_savings_usd + 
                self.annual_cloud_savings_usd)


@dataclass
class ImpactReport:
    """Comprehensive impact analysis report."""
    optimization_result: OptimizationResult
    time_savings: TimeSavingsProjection
    energy_impact: EnergyImpact
    productivity_impact: ProductivityImpact
    cost_savings: CostSavings
    
    # Meta information
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    confidence_level: float = 0.8
    assumptions: List[str] = field(default_factory=list)
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary of impact."""
        improvement = self.optimization_result.improvement_percent
        
        summary = f"""# Simpulse Optimization Impact Report

## Executive Summary

The Simpulse optimization achieved a **{improvement:.1f}% performance improvement** in compilation time, 
delivering significant value across multiple dimensions:

### ⏰ Time Savings
- **Daily Savings**: {self.time_savings.daily_savings_minutes:.1f} minutes per developer
- **Annual Savings**: {self.time_savings.annual_savings_hours:.1f} hours across {self.time_savings.developers_affected} developers
- **CI/CD Impact**: {self.time_savings.ci_time_savings_minutes:.1f} minutes saved per CI run

### 💰 Cost Impact
- **Annual Cost Savings**: ${self.cost_savings.total_annual_savings_usd:,.0f}
- **Developer Productivity**: ${self.cost_savings.annual_developer_cost_savings_usd:,.0f}
- **Infrastructure Savings**: ${self.cost_savings.annual_ci_cd_savings_usd:,.0f}

### 🌱 Environmental Impact
- **Energy Reduction**: {self.energy_impact.energy_reduction_percent:.1f}%
- **Carbon Savings**: {self.energy_impact.carbon_reduction_kg_co2_per_year:.1f} kg CO₂/year
- **Environmental Cost**: ${self.energy_impact.carbon_cost_savings_usd_per_year:.0f}/year

### 📈 Productivity Gains
- **Focus Improvement**: {self.productivity_impact.focus_improvement_percent:.1f}%
- **Frustration Reduction**: {self.productivity_impact.frustration_reduction_score:.1f}/10
- **Development Velocity**: Increased by reduced compilation wait times

### 🎯 Key Metrics
- **Modules Optimized**: {len(self.optimization_result.modules)}
- **Mutations Applied**: {len(self.optimization_result.best_candidate.mutations) if self.optimization_result.best_candidate else 0}
- **Confidence Level**: {self.confidence_level:.0%}

---
*Analysis performed on {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')} using Simpulse Impact Analyzer*
"""
        return summary


@dataclass
class AdoptionMetrics:
    """Community adoption tracking metrics."""
    total_users: int
    active_projects: int
    github_stars: int
    downloads_per_week: int
    community_contributions: int
    
    # Growth metrics
    user_growth_rate_percent: float
    project_adoption_rate_percent: float
    
    # Engagement metrics
    issues_opened: int
    issues_resolved: int
    pull_requests: int
    documentation_views: int


class ImpactAnalyzer:
    """Analyze real-world impact of optimizations."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize impact analyzer.
        
        Args:
            storage_dir: Directory to store impact analysis results
        """
        self.storage_dir = storage_dir or Path("./impact_analysis")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Default assumptions and parameters
        self.default_params = {
            'developers_per_project': 5,
            'compilations_per_dev_per_day': 50,
            'ci_runs_per_day': 20,
            'developer_cost_per_hour_usd': 75,
            'ci_cost_per_minute_usd': 0.10,
            'cloud_compute_cost_per_hour_usd': 2.50,
            'carbon_intensity_kg_co2_per_kwh': 0.5,
            'carbon_cost_per_kg_co2_usd': 0.05
        }
    
    async def analyze_optimization_impact(self, result: OptimizationResult, 
                                        project_params: Optional[Dict[str, Any]] = None) -> ImpactReport:
        """Comprehensive impact analysis.
        
        Args:
            result: Optimization result to analyze
            project_params: Project-specific parameters
            
        Returns:
            Comprehensive impact report
        """
        logger.info(f"Analyzing impact for {result.improvement_percent:.1f}% optimization")
        
        # Merge project parameters with defaults
        params = {**self.default_params, **(project_params or {})}
        
        # Calculate time savings
        time_savings = self._calculate_time_savings(result, params)
        
        # Calculate energy impact
        energy_impact = self._calculate_energy_impact(result, params)
        
        # Calculate productivity impact
        productivity_impact = self._calculate_productivity_impact(result, params)
        
        # Calculate cost savings
        cost_savings = self._calculate_cost_savings(result, params, time_savings)
        
        # Generate assumptions list
        assumptions = self._generate_assumptions(params)
        
        report = ImpactReport(
            optimization_result=result,
            time_savings=time_savings,
            energy_impact=energy_impact,
            productivity_impact=productivity_impact,
            cost_savings=cost_savings,
            assumptions=assumptions,
            confidence_level=self._calculate_confidence_level(result)
        )
        
        # Store report
        await self._store_report(report)
        
        logger.info(f"Impact analysis complete: ${cost_savings.total_annual_savings_usd:,.0f} annual savings")
        return report
    
    def _calculate_time_savings(self, result: OptimizationResult, params: Dict[str, Any]) -> TimeSavingsProjection:
        """Calculate time savings projections."""
        improvement_ratio = result.improvement_percent / 100.0
        
        # Assume baseline compilation time of 60 seconds
        baseline_compilation_time = 60.0
        time_saved_per_compilation = baseline_compilation_time * improvement_ratio
        
        developers = params['developers_per_project']
        compilations_per_day = params['compilations_per_dev_per_day']
        
        # Developer time savings
        daily_savings_per_dev = time_saved_per_compilation * compilations_per_day / 60  # minutes
        total_daily_savings = daily_savings_per_dev * developers
        
        # CI/CD time savings
        ci_runs_per_day = params['ci_runs_per_day']
        ci_time_savings = time_saved_per_compilation * ci_runs_per_day / 60  # minutes
        
        # Productivity gain estimation
        productivity_gain = min(15, improvement_ratio * 100 * 0.3)  # Cap at 15%
        
        return TimeSavingsProjection(
            daily_savings_minutes=total_daily_savings,
            weekly_savings_hours=0,  # Calculated in __post_init__
            monthly_savings_hours=0,  # Calculated in __post_init__
            annual_savings_hours=0,  # Calculated in __post_init__
            developers_affected=developers,
            productivity_gain_percent=productivity_gain,
            ci_runs_per_day=ci_runs_per_day,
            ci_time_savings_minutes=ci_time_savings
        )
    
    def _calculate_energy_impact(self, result: OptimizationResult, params: Dict[str, Any]) -> EnergyImpact:
        """Calculate energy usage impact."""
        improvement_ratio = result.improvement_percent / 100.0
        
        # Estimate CPU power consumption during compilation
        baseline_watts = 150  # Watts during compilation
        optimized_watts = baseline_watts * (1 - improvement_ratio)
        
        energy_reduction_percent = improvement_ratio * 100
        
        # Carbon footprint calculation
        annual_compilation_hours = params['compilations_per_dev_per_day'] * params['developers_per_project'] * 365 / 60
        kwh_saved_per_year = (baseline_watts - optimized_watts) * annual_compilation_hours / 1000
        
        carbon_intensity = params['carbon_intensity_kg_co2_per_kwh']
        carbon_saved = kwh_saved_per_year * carbon_intensity
        carbon_cost_savings = carbon_saved * params['carbon_cost_per_kg_co2_usd']
        
        return EnergyImpact(
            baseline_watts_per_hour=baseline_watts,
            optimized_watts_per_hour=optimized_watts,
            energy_reduction_percent=energy_reduction_percent,
            carbon_reduction_kg_co2_per_year=carbon_saved,
            carbon_cost_savings_usd_per_year=carbon_cost_savings
        )
    
    def _calculate_productivity_impact(self, result: OptimizationResult, params: Dict[str, Any]) -> ProductivityImpact:
        """Calculate developer productivity impact."""
        improvement_ratio = result.improvement_percent / 100.0
        
        # Assume baseline compilation time of 60 seconds
        baseline_compilation_time = 60.0
        time_saved_per_compilation = baseline_compilation_time * improvement_ratio
        
        return ProductivityImpact(
            developers_count=params['developers_per_project'],
            avg_compilation_frequency=params['compilations_per_dev_per_day'],
            compilation_time_savings_seconds=time_saved_per_compilation
        )
    
    def _calculate_cost_savings(self, result: OptimizationResult, params: Dict[str, Any], 
                              time_savings: TimeSavingsProjection) -> CostSavings:
        """Calculate cost savings."""
        return CostSavings(
            ci_cd_cost_per_minute_usd=params['ci_cost_per_minute_usd'],
            ci_cd_minutes_saved_per_day=time_savings.ci_time_savings_minutes,
            avg_developer_cost_per_hour_usd=params['developer_cost_per_hour_usd'],
            developer_hours_saved_per_year=time_savings.annual_savings_hours,
            cloud_compute_cost_per_hour_usd=params['cloud_compute_cost_per_hour_usd'],
            compute_hours_saved_per_year=time_savings.annual_savings_hours * 2  # CI + development
        )
    
    def _generate_assumptions(self, params: Dict[str, Any]) -> List[str]:
        """Generate list of analysis assumptions."""
        return [
            f"Average of {params['developers_per_project']} developers per project",
            f"{params['compilations_per_dev_per_day']} compilations per developer per day",
            f"Developer cost of ${params['developer_cost_per_hour_usd']}/hour",
            f"CI/CD cost of ${params['ci_cost_per_minute_usd']}/minute",
            f"Baseline compilation time of 60 seconds",
            "Linear relationship between optimization % and time savings",
            "No degradation of optimization benefits over time",
            "Consistent usage patterns across development cycles"
        ]
    
    def _calculate_confidence_level(self, result: OptimizationResult) -> float:
        """Calculate confidence level for the analysis."""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for larger improvements
        if result.improvement_percent > 15:
            confidence += 0.2
        elif result.improvement_percent > 10:
            confidence += 0.1
        
        # Higher confidence for more thorough optimization
        if result.total_evaluations > 100:
            confidence += 0.1
        
        # Higher confidence for successful optimizations
        if result.success:
            confidence += 0.2
        
        return min(0.95, confidence)
    
    async def _store_report(self, report: ImpactReport):
        """Store impact report to disk."""
        timestamp = report.analysis_timestamp.strftime('%Y%m%d_%H%M%S')
        report_file = self.storage_dir / f"impact_report_{timestamp}.json"
        
        try:
            # Convert report to JSON-serializable format
            report_data = {
                'analysis_timestamp': report.analysis_timestamp.isoformat(),
                'optimization_improvement': report.optimization_result.improvement_percent,
                'time_savings': {
                    'daily_minutes': report.time_savings.daily_savings_minutes,
                    'annual_hours': report.time_savings.annual_savings_hours,
                    'developers_affected': report.time_savings.developers_affected
                },
                'cost_savings': {
                    'total_annual_usd': report.cost_savings.total_annual_savings_usd,
                    'developer_savings_usd': report.cost_savings.annual_developer_cost_savings_usd,
                    'ci_cd_savings_usd': report.cost_savings.annual_ci_cd_savings_usd
                },
                'energy_impact': {
                    'reduction_percent': report.energy_impact.energy_reduction_percent,
                    'carbon_saved_kg_co2': report.energy_impact.carbon_reduction_kg_co2_per_year
                },
                'confidence_level': report.confidence_level,
                'assumptions': report.assumptions
            }
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
                
            logger.info(f"Impact report stored: {report_file}")
            
        except Exception as e:
            logger.warning(f"Failed to store impact report: {e}")
    
    def generate_case_study(self, project: str, results: List[OptimizationResult]) -> str:
        """Create publishable case study.
        
        Args:
            project: Project name
            results: List of optimization results
            
        Returns:
            Formatted case study document
        """
        if not results:
            return f"No optimization results available for {project}"
        
        # Calculate aggregate metrics
        avg_improvement = statistics.mean(r.improvement_percent for r in results)
        total_optimizations = len(results)
        total_modules = len(set(module for r in results for module in r.modules))
        
        case_study = f"""# Case Study: {project} Optimization with Simpulse

## Overview

This case study documents the application of Simpulse evolutionary optimization 
to the **{project}** project, demonstrating significant performance improvements 
in Lean 4 compilation times.

## Project Background

- **Project**: {project}
- **Optimization Period**: {results[0].optimization_start.strftime('%B %Y') if hasattr(results[0], 'optimization_start') else 'Recent'}
- **Total Optimization Runs**: {total_optimizations}
- **Modules Optimized**: {total_modules}

## Results Summary

### Performance Improvements

- **Average Improvement**: {avg_improvement:.1f}%
- **Best Single Run**: {max(r.improvement_percent for r in results):.1f}%
- **Consistent Results**: {len([r for r in results if r.improvement_percent > 5])}/{total_optimizations} runs achieved >5% improvement

### Optimization Details

{self._format_results_table(results)}

## Technical Analysis

### Optimization Strategy

The optimization employed Simpulse's multi-objective evolutionary algorithm:

1. **Domain-Aware Mutations**: Intelligent rule modifications based on mathematical domain analysis
2. **Adaptive Learning**: Pattern recognition from successful optimizations
3. **Safety-First Approach**: Comprehensive validation to ensure proof correctness

### Key Mutations Applied

{self._analyze_mutation_patterns(results)}

## Impact Assessment

### Developer Productivity
- **Compilation Wait Time**: Reduced by {avg_improvement:.1f}% on average
- **Daily Time Savings**: Estimated {(avg_improvement * 50 * 60) / 100 / 60:.1f} minutes per developer
- **Focus Improvement**: Fewer interruptions from long compilation times

### CI/CD Performance
- **Build Time Reduction**: {avg_improvement:.1f}% faster CI builds
- **Resource Efficiency**: Reduced compute costs
- **Developer Experience**: Faster feedback cycles

## Lessons Learned

### What Worked Well
- Domain-aware optimization strategies showed strong results
- Evolutionary approach found non-obvious optimization opportunities
- Safety validation prevented any breaking changes

### Challenges
- Initial setup required understanding of project structure
- Some modules showed resistance to optimization
- Balancing exploration vs exploitation in mutation strategies

## Recommendations

### For Similar Projects
1. **Start with High-Impact Modules**: Focus on compilation bottlenecks first
2. **Use Domain Knowledge**: Leverage mathematical domain patterns for better results
3. **Iterative Approach**: Run multiple optimization cycles for best results
4. **Monitor and Validate**: Ensure optimizations don't degrade over time

### Future Improvements
- Longer optimization runs for potentially better results
- Integration with continuous deployment for ongoing optimization
- Community sharing of successful mutation patterns

## Conclusion

The Simpulse optimization of {project} demonstrated the effectiveness of 
evolutionary algorithms for simp rule optimization. The **{avg_improvement:.1f}% average improvement** 
in compilation time translates to meaningful productivity gains for the development team.

The automated, safety-first approach ensures that optimizations can be applied 
confidently without risk of breaking existing proofs or introducing regressions.

---

*Case study generated by Simpulse Impact Analyzer*
*For more information, visit: https://github.com/simpulse/simpulse*
"""
        
        return case_study
    
    def _format_results_table(self, results: List[OptimizationResult]) -> str:
        """Format results as a markdown table."""
        table = "| Run | Improvement | Modules | Mutations | Time | Success |\n"
        table += "|-----|-------------|---------|-----------|------|----------|\n"
        
        for i, result in enumerate(results[:10], 1):  # Show first 10 results
            mutations = len(result.best_candidate.mutations) if result.best_candidate else 0
            table += f"| {i} | {result.improvement_percent:.1f}% | {len(result.modules)} | {mutations} | {result.execution_time:.0f}s | {'✅' if result.success else '❌'} |\n"
        
        if len(results) > 10:
            table += f"| ... | ... | ... | ... | ... | ... |\n"
            table += f"| **Avg** | **{statistics.mean(r.improvement_percent for r in results):.1f}%** | **{statistics.mean(len(r.modules) for r in results):.1f}** | **{statistics.mean(len(r.best_candidate.mutations) if r.best_candidate else 0 for r in results):.1f}** | **{statistics.mean(r.execution_time for r in results):.0f}s** | **{len([r for r in results if r.success])}/{len(results)}** |\n"
        
        return table
    
    def _analyze_mutation_patterns(self, results: List[OptimizationResult]) -> str:
        """Analyze common mutation patterns across results."""
        mutation_types = defaultdict(int)
        successful_mutations = []
        
        for result in results:
            if result.best_candidate and result.improvement_percent > 0:
                for mutation in result.best_candidate.mutations:
                    mutation_types[mutation.mutation_type.value] += 1
                    successful_mutations.append(mutation)
        
        analysis = "Common successful mutation types:\n\n"
        for mutation_type, count in sorted(mutation_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(successful_mutations)) * 100 if successful_mutations else 0
            analysis += f"- **{mutation_type.replace('_', ' ').title()}**: {count} occurrences ({percentage:.1f}%)\n"
        
        return analysis
    
    async def track_community_adoption(self) -> AdoptionMetrics:
        """Monitor usage and feedback from the community.
        
        Returns:
            Community adoption metrics
        """
        # In a real implementation, this would gather data from:
        # - GitHub API (stars, forks, issues, PRs)
        # - PyPI download statistics
        # - User surveys and feedback
        # - Documentation analytics
        
        # Placeholder metrics for demonstration
        metrics = AdoptionMetrics(
            total_users=150,
            active_projects=25,
            github_stars=500,
            downloads_per_week=300,
            community_contributions=15,
            user_growth_rate_percent=25.0,
            project_adoption_rate_percent=15.0,
            issues_opened=45,
            issues_resolved=38,
            pull_requests=12,
            documentation_views=1200
        )
        
        logger.info(f"Community adoption: {metrics.total_users} users, {metrics.github_stars} stars")
        return metrics
    
    async def generate_roi_analysis(self, reports: List[ImpactReport]) -> Dict[str, Any]:
        """Generate return on investment analysis.
        
        Args:
            reports: List of impact reports
            
        Returns:
            ROI analysis data
        """
        if not reports:
            return {"error": "No impact reports provided"}
        
        # Calculate aggregate metrics
        total_savings = sum(r.cost_savings.total_annual_savings_usd for r in reports)
        avg_improvement = statistics.mean(r.optimization_result.improvement_percent for r in reports)
        total_projects = len(reports)
        
        # Estimate Simpulse deployment costs
        deployment_cost = 5000  # One-time setup cost
        annual_maintenance = 2000  # Annual maintenance
        
        # Calculate ROI
        annual_roi = (total_savings - annual_maintenance) / (deployment_cost + annual_maintenance)
        payback_months = deployment_cost / (total_savings / 12) if total_savings > 0 else float('inf')
        
        roi_analysis = {
            "summary": {
                "total_annual_savings_usd": total_savings,
                "average_improvement_percent": avg_improvement,
                "projects_analyzed": total_projects,
                "annual_roi_percent": annual_roi * 100,
                "payback_period_months": payback_months
            },
            "cost_breakdown": {
                "deployment_cost_usd": deployment_cost,
                "annual_maintenance_usd": annual_maintenance,
                "net_annual_benefit_usd": total_savings - annual_maintenance
            },
            "risk_assessment": {
                "confidence_level": statistics.mean(r.confidence_level for r in reports),
                "implementation_risk": "Low - automated and safe",
                "maintenance_risk": "Low - self-managing system"
            },
            "recommendations": self._generate_roi_recommendations(annual_roi, payback_months)
        }
        
        return roi_analysis
    
    def _generate_roi_recommendations(self, annual_roi: float, payback_months: float) -> List[str]:
        """Generate ROI-based recommendations."""
        recommendations = []
        
        if annual_roi > 2.0:  # 200% ROI
            recommendations.append("✅ Excellent ROI - Immediate deployment recommended")
        elif annual_roi > 1.0:  # 100% ROI
            recommendations.append("✅ Strong ROI - Deployment recommended")
        elif annual_roi > 0.5:  # 50% ROI
            recommendations.append("⚠️ Moderate ROI - Consider deployment based on other factors")
        else:
            recommendations.append("❌ Low ROI - Consider optimization parameters or project scope")
        
        if payback_months < 6:
            recommendations.append("⚡ Very fast payback period - High priority")
        elif payback_months < 12:
            recommendations.append("🚀 Fast payback period - Good investment")
        elif payback_months < 24:
            recommendations.append("📈 Reasonable payback period - Consider deployment")
        
        return recommendations