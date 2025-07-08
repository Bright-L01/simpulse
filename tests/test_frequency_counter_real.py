"""
Test the REAL frequency counter functionality.
This is one of the few components that actually works.
"""

import tempfile
from pathlib import Path

from simpulse.analysis.frequency_counter import FrequencyCounter


class TestFrequencyCounter:
    """Test real trace parsing functionality."""

    def test_parse_success_trace(self):
        """Test parsing successful simp application."""
        counter = FrequencyCounter()

        trace_line = "[trace.Meta.Tactic.simp.rewrite] Nat.add_zero: n + 0 ==> n"

        matches = counter._parse_trace_line(trace_line)
        assert len(matches) > 0
        assert matches[0]["lemma"] == "Nat.add_zero"
        assert matches[0]["type"] == "success"

    def test_parse_try_lemma_trace(self):
        """Test parsing simp lemma attempts."""
        counter = FrequencyCounter()

        trace_line = "[trace.Meta.Tactic.simp] trying simp lemma Nat.mul_one"

        matches = counter._parse_trace_line(trace_line)
        assert len(matches) > 0
        assert matches[0]["lemma"] == "Nat.mul_one"
        assert matches[0]["type"] == "try"

    def test_count_frequencies_from_trace(self):
        """Test frequency counting from complete trace."""
        counter = FrequencyCounter()

        trace_content = """
[trace.Meta.Tactic.simp.rewrite] Nat.add_zero: n + 0 ==> n
[trace.Meta.Tactic.simp] trying simp lemma Nat.mul_one
[trace.Meta.Tactic.simp.rewrite] Nat.mul_one: n * 1 ==> n
[trace.Meta.Tactic.simp] trying simp lemma Nat.add_zero
[trace.Meta.Tactic.simp.rewrite] Nat.add_zero: m + 0 ==> m
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".trace", delete=False) as f:
            f.write(trace_content)
            trace_file = Path(f.name)

        try:
            frequencies = counter.count_frequencies(trace_file)

            assert "Nat.add_zero" in frequencies
            assert frequencies["Nat.add_zero"]["success_count"] == 2
            assert frequencies["Nat.add_zero"]["try_count"] == 1

            assert "Nat.mul_one" in frequencies
            assert frequencies["Nat.mul_one"]["success_count"] == 1
            assert frequencies["Nat.mul_one"]["try_count"] == 1
        finally:
            trace_file.unlink()

    def test_handle_different_trace_formats(self):
        """Test handling various trace format variations."""
        counter = FrequencyCounter()

        trace_variants = [
            "[trace.Tactic.simp.rewrite] foo: x ==> y",
            "[trace.Meta.Tactic.simp.rewrite] bar: a ==>simp b",
            "[trace.Meta.Tactic.simp] apply Nat.zero_add",
            "[trace.Meta.Tactic.simp] trying lemma List.map_nil",
        ]

        for trace in trace_variants:
            matches = counter._parse_trace_line(trace)
            assert len(matches) > 0, f"Failed to parse: {trace}"

    def test_frequency_to_priority_conversion(self):
        """Test converting frequencies to priority assignments."""
        counter = FrequencyCounter()

        frequencies = {
            "Nat.add_zero": {
                "success_count": 100,
                "try_count": 100,
                "total_count": 200,
                "success_rate": 1.0,
            },
            "Nat.mul_one": {
                "success_count": 50,
                "try_count": 60,
                "total_count": 110,
                "success_rate": 0.833,
            },
            "rare_lemma": {
                "success_count": 1,
                "try_count": 5,
                "total_count": 6,
                "success_rate": 0.2,
            },
        }

        # Get top lemmas for optimization
        top_lemmas = counter.get_optimization_candidates(frequencies, top_n=2)

        assert len(top_lemmas) == 2
        assert top_lemmas[0][0] == "Nat.add_zero"
        assert top_lemmas[1][0] == "Nat.mul_one"

    def test_empty_trace_handling(self):
        """Test handling of empty or invalid traces."""
        counter = FrequencyCounter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".trace", delete=False) as f:
            f.write("")
            trace_file = Path(f.name)

        try:
            frequencies = counter.count_frequencies(trace_file)
            assert frequencies == {}
        finally:
            trace_file.unlink()

    def test_malformed_trace_lines(self):
        """Test robustness against malformed trace lines."""
        counter = FrequencyCounter()

        malformed_lines = [
            "Not a trace line",
            "[trace.simp] ",
            "[trace.Meta.Tactic.simp.rewrite]",
            "trying simp lemma without trace prefix",
        ]

        for line in malformed_lines:
            matches = counter._parse_trace_line(line)
            # Should handle gracefully without crashing
            assert isinstance(matches, list)

    def test_hot_spot_detection(self):
        """Test detection of hot spots in traces."""
        counter = FrequencyCounter()

        trace_content = """
-- File: test.lean:10
[trace.Meta.Tactic.simp.rewrite] Nat.add_zero: n + 0 ==> n
[trace.Meta.Tactic.simp.rewrite] Nat.add_zero: m + 0 ==> m
[trace.Meta.Tactic.simp.rewrite] Nat.add_zero: k + 0 ==> k
-- File: other.lean:20
[trace.Meta.Tactic.simp.rewrite] Nat.mul_one: x * 1 ==> x
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".trace", delete=False) as f:
            f.write(trace_content)
            trace_file = Path(f.name)

        try:
            hot_spots = counter.find_hot_spots(trace_file, threshold=3)

            # Should detect test.lean:10 as hot spot
            assert len(hot_spots) > 0
            assert any("test.lean" in str(spot) for spot in hot_spots)
        finally:
            trace_file.unlink()

    def test_optimization_impact_estimation(self):
        """Test estimation of optimization impact."""
        counter = FrequencyCounter()

        frequencies = {
            "Nat.add_zero": {
                "success_count": 1000,
                "try_count": 1000,
                "total_count": 2000,
                "success_rate": 1.0,
            },
            "rarely_used": {
                "success_count": 1,
                "try_count": 10,
                "total_count": 11,
                "success_rate": 0.1,
            },
        }

        # Estimate impact of optimizing top lemmas
        impact = counter.estimate_optimization_impact(frequencies, top_n=1)

        assert impact is not None
        assert impact["estimated_reduction_percent"] > 0
        assert "Nat.add_zero" in impact["optimized_lemmas"]
