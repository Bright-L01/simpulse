"""
Tests for report generation functionality.
"""

from unittest.mock import Mock, patch

import pytest

from simpulse.evolution.evolution_engine import OptimizationResult
from simpulse.evolution.models_v2 import (
    Candidate,
    EvolutionHistory,
    FitnessScore,
    GenerationResult,
)
from simpulse.reporting.report_generator import (
    PLOTLY_AVAILABLE,
    Dashboard,
    ReportGenerator,
)


class TestReportGenerator:
    """Test suite for ReportGenerator."""

    @pytest.fixture
    def generator(self):
        """Create a ReportGenerator instance for testing."""
        return ReportGenerator()

    @pytest.fixture
    def sample_result(self):
        """Create a sample optimization result for testing."""
        # Create fitness score
        fitness = FitnessScore(
            time_score=0.8,
            memory_score=0.7,
            iterations_score=0.9,
            depth_score=0.85,
            composite_score=0.8125,
            total_time=40.0,
            simp_time=8.0,
            memory_mb=200.0,
            iterations=85,
            depth=4,
        )

        # Create best candidate
        best_candidate = Candidate(mutations=[], fitness=fitness)

        # Create history
        history = EvolutionHistory()
        for i in range(5):
            gen_result = GenerationResult(
                generation=i,
                best_fitness=0.6 + i * 0.05,
                average_fitness=0.5 + i * 0.04,
                diversity_score=0.8 - i * 0.1,
                valid_candidates=20 - i,
                evaluation_time=10.0 + i,
            )
            history.add_generation(gen_result)

        # Create optimization result
        return OptimizationResult(
            success=True,
            modules=["TestModule1", "TestModule2"],
            best_candidate=best_candidate,
            improvement_percent=20.0,
            total_generations=5,
            total_evaluations=100,
            execution_time=60.0,
            history=history,
        )

    @pytest.mark.asyncio
    async def test_generate_html_report(self, generator, sample_result):
        """Test HTML report generation."""
        html = await generator.generate_html_report(sample_result)

        # Verify HTML structure
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html

        # Check content sections
        assert "Simpulse Optimization Report" in html
        assert "Executive Summary" in html
        assert "Performance Analysis" in html
        assert "20.0% Improvement" in html
        assert "TestModule1" in html
        assert "TestModule2" in html

    @pytest.mark.asyncio
    async def test_generate_html_without_plotly(self, generator, sample_result):
        """Test HTML report generation without Plotly."""
        # Mock plotly as unavailable
        with patch.object(generator, "plotly_available", False):
            html = await generator.generate_html_report(
                sample_result, include_interactive=True
            )

            # Should still generate valid HTML
            assert isinstance(html, str)
            assert "<!DOCTYPE html>" in html
            # Should not have plotly scripts
            assert "plotly" not in html.lower() or "not available" in html

    def test_generate_markdown_summary(self, generator, sample_result):
        """Test Markdown report generation."""
        markdown = generator.generate_markdown_summary(sample_result)

        # Verify Markdown structure
        assert isinstance(markdown, str)
        assert "# 🧬 Simpulse Optimization Report" in markdown
        assert "## 📊 Summary" in markdown
        assert "| Metric | Value |" in markdown
        assert "| **Performance Improvement** | 20.0% |" in markdown
        assert "TestModule1" in markdown

    def test_generate_performance_section(self, generator, sample_result):
        """Test performance section generation."""
        section = generator._generate_performance_section(sample_result)

        assert "Performance Analysis" in section
        assert "<table" in section
        assert "Baseline" in section
        assert "Optimized" in section
        assert "Improvement" in section
        assert "40.00 ms" in section  # Total time
        assert "8.00 ms" in section  # Simp time

    def test_generate_mutations_section(self, generator):
        """Test mutations section generation."""
        # Create result with mutations
        from simpulse.evolution.models import MutationSuggestion, MutationType

        mutation = Mock()
        mutation.suggestion = MutationSuggestion(
            rule_name="test_rule",
            mutation_type=MutationType.PRIORITY_CHANGE,
            confidence=0.85,
            description="Increase priority to improve performance",
            estimated_impact={"time": -15.0, "memory": -5.0},
        )

        candidate = Candidate(mutations=[mutation])
        result = OptimizationResult(
            success=True,
            modules=["Test"],
            best_candidate=candidate,
            improvement_percent=15.0,
        )

        section = generator._generate_mutations_section(result)

        assert "Applied Mutations" in section
        assert "test_rule" in section
        assert "PRIORITY_CHANGE" in section
        assert "85.0%" in section
        assert "Increase priority" in section

    def test_generate_evolution_section(self, generator, sample_result):
        """Test evolution statistics section generation."""
        section = generator._generate_evolution_section(sample_result)

        assert "Evolution Statistics" in section
        assert "Total Evaluations:" in section
        assert "100" in section
        assert "Success Rate:" in section
        assert "True" in section

    def test_css_styles(self, generator):
        """Test CSS styles generation."""
        css = generator._get_css_styles()

        assert "body {" in css
        assert "font-family:" in css
        assert ".container {" in css
        assert ".metric-card {" in css
        assert "color:" in css
        assert "background:" in css

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_create_performance_timeline(self, generator, sample_result):
        """Test performance timeline plot creation."""
        fig = generator._create_performance_timeline(sample_result.history)

        # Verify figure structure
        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) >= 2  # At least best and average fitness

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_create_fitness_distribution(self, generator, sample_result):
        """Test fitness distribution plot creation."""
        fig = generator._create_fitness_distribution(sample_result.history)

        assert fig is not None
        assert hasattr(fig, "data")

    def test_create_performance_dashboard(self, generator, sample_result):
        """Test dashboard creation."""
        dashboard = generator.create_performance_dashboard(sample_result.history)

        assert isinstance(dashboard, Dashboard)
        assert len(dashboard.generation_data) == 5
        assert dashboard.mutation_stats["convergence_generation"] is None
        assert dashboard.resource_usage["cpu_time_seconds"] > 0

    def test_format_mutation_impact(self, generator):
        """Test mutation impact formatting."""
        impact = {"time": -10.0, "memory": -5.0, "iterations": 15.0}

        html = generator._format_mutation_impact(impact)

        assert "time: -10.0%" in html
        assert "memory: -5.0%" in html
        assert "iterations: 15.0%" in html
        assert '<div class="impact-list">' in html

    def test_empty_result_handling(self, generator):
        """Test handling of empty optimization results."""
        empty_result = OptimizationResult(
            success=False, modules=[], improvement_percent=0.0
        )

        # Should handle gracefully
        markdown = generator.generate_markdown_summary(empty_result)
        assert "0.0%" in markdown

        section = generator._generate_performance_section(empty_result)
        assert "No performance data available" in section

    @pytest.mark.asyncio
    async def test_save_report_to_file(self, generator, sample_result, temp_dir):
        """Test saving report to file."""
        output_file = temp_dir / "report.html"

        html = await generator.generate_html_report(sample_result)

        # Save to file
        with open(output_file, "w") as f:
            f.write(html)

        # Verify file was created
        assert output_file.exists()
        assert output_file.stat().st_size > 0

        # Verify content
        content = output_file.read_text()
        assert "Simpulse Optimization Report" in content


class TestDashboard:
    """Test suite for Dashboard class."""

    def test_dashboard_initialization(self):
        """Test Dashboard initialization."""
        dashboard = Dashboard()

        assert isinstance(dashboard.generation_data, list)
        assert isinstance(dashboard.fitness_timeline, list)
        assert isinstance(dashboard.diversity_timeline, list)
        assert isinstance(dashboard.mutation_stats, dict)
        assert isinstance(dashboard.resource_usage, dict)

    def test_add_generation_data(self):
        """Test adding generation data to dashboard."""
        dashboard = Dashboard()

        gen_result = GenerationResult(
            generation=0,
            best_fitness=0.8,
            average_fitness=0.6,
            diversity_score=0.7,
            valid_candidates=20,
        )

        dashboard.add_generation_data(gen_result)

        assert len(dashboard.generation_data) == 1
        assert dashboard.generation_data[0]["generation"] == 0
        assert dashboard.generation_data[0]["best_fitness"] == 0.8
        assert "timestamp" in dashboard.generation_data[0]

    def test_dashboard_to_dict(self):
        """Test converting dashboard to dictionary."""
        dashboard = Dashboard()

        # Add some data
        dashboard.mutation_stats = {"total": 10, "successful": 8}
        dashboard.resource_usage = {"cpu": 45.0, "memory": 1024.0}

        data = dashboard.to_dict()

        assert isinstance(data, dict)
        assert "generation_data" in data
        assert "mutation_stats" in data
        assert data["mutation_stats"]["total"] == 10
        assert data["resource_usage"]["cpu"] == 45.0


@pytest.mark.integration
class TestReportGeneratorIntegration:
    """Integration tests for report generator."""

    @pytest.mark.asyncio
    async def test_full_report_generation(self, temp_dir):
        """Test complete report generation workflow."""
        generator = ReportGenerator()

        # Create comprehensive result
        result = OptimizationResult(
            success=True,
            modules=["Module1", "Module2", "Module3"],
            improvement_percent=25.5,
            total_generations=10,
            total_evaluations=200,
            execution_time=120.0,
        )

        # Generate reports
        html = await generator.generate_html_report(result)
        markdown = generator.generate_markdown_summary(result)

        # Save reports
        html_file = temp_dir / "full_report.html"
        md_file = temp_dir / "summary.md"

        with open(html_file, "w") as f:
            f.write(html)

        with open(md_file, "w") as f:
            f.write(markdown)

        # Verify files
        assert html_file.exists()
        assert md_file.exists()
        assert html_file.stat().st_size > 1000  # Should be substantial
        assert md_file.stat().st_size > 100
