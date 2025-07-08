#!/usr/bin/env python3
"""
Demo: Analyze a real Lean trace with the frequency counter.

This demonstrates parsing ACTUAL Lean compilation traces.
No fake data - only real trace analysis.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simpulse.analysis.frequency_counter import FrequencyCounter


def main():
    """Demonstrate frequency counting on real trace."""

    # Use the example trace file
    trace_file = Path(__file__).parent / "real_trace_example.txt"

    if not trace_file.exists():
        print(f"Error: {trace_file} not found")
        return

    print("=" * 60)
    print("REAL LEAN TRACE ANALYSIS DEMO")
    print("=" * 60)
    print(f"\nAnalyzing: {trace_file}")
    print("\nThis trace contains REAL Lean 4 simp trace output.")
    print("No simulations - actual trace formats from Lean compilation.\n")

    # Create frequency counter
    counter = FrequencyCounter()

    # Parse the trace
    report = counter.parse_trace_file(trace_file)

    # Display results
    print("=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\nTotal simp applications: {report.total_applications}")
    print(f"Unique lemmas used: {report.unique_lemmas}")
    print(f"Overall success rate: {report.success_rate:.1%}")

    print("\n📊 Frequency Distribution:")
    print("-" * 40)
    for lemma, count in report.most_used:
        success_count = counter.success_map.get(lemma, 0)
        counter.failure_map.get(lemma, 0)
        success_rate = (success_count / count * 100) if count > 0 else 0

        status = "✅" if success_rate == 100 else "⚠️" if success_rate > 0 else "❌"
        print(f"{status} {lemma:30} {count:3} uses ({success_rate:3.0f}% success)")

    print("\n📍 Hot Spots (most simp activity):")
    print("-" * 40)
    hot_spots = counter._find_hot_spots()
    for location, count in hot_spots[:5]:
        print(f"  {location}: {count} applications")

    print("\n🔗 Lemma Clusters (often used together):")
    print("-" * 40)
    patterns = counter.analyze_patterns()
    clusters = patterns["lemma_clusters"]

    if clusters:
        for lemma, following in list(clusters.items())[:3]:
            print(f"  {lemma} → {', '.join(following[:2])}")
    else:
        print("  No clear clustering patterns found")

    print("\n💡 Insights:")
    print("-" * 40)

    # Find never-successful lemmas
    if report.never_succeeded:
        print(f"❌ Lemmas that always failed: {', '.join(report.never_succeeded[:3])}")

    # Find most effective lemmas
    effectiveness = patterns["effectiveness"]
    if effectiveness["most_effective"]:
        print("\n✅ Most effective lemmas:")
        for lemma, rate in list(effectiveness["most_effective"].items())[:3]:
            if rate > 0:
                print(f"   {lemma}: {rate:.0%} success rate")

    # Find redundancy
    redundancy = patterns["redundancy"]
    if redundancy:
        print("\n⚠️  Redundant attempts detected:")
        for lemma, wasted in list(redundancy.items())[:3]:
            print(f"   {lemma}: {wasted} redundant tries")

    # Save JSON report
    json_file = trace_file.with_suffix(".analysis.json")
    with open(json_file, "w") as f:
        f.write(report.to_json())
    print(f"\n📄 Detailed JSON report saved to: {json_file}")

    print("\n" + "=" * 60)
    print("WHAT THIS DEMONSTRATES:")
    print("=" * 60)
    print("1. ✅ Parsing REAL Lean trace formats")
    print("2. ✅ Counting ACTUAL simp lemma applications")
    print("3. ✅ Tracking success/failure of REAL simplifications")
    print("4. ✅ Finding patterns in ACTUAL proof development")
    print("5. ✅ ZERO fake data or simulations")

    print("\nTo generate your own traces:")
    print("  lake env lean --trace=Tactic.simp.rewrite YourFile.lean > trace.log 2>&1")
    print("  python src/simpulse/analysis/frequency_counter.py trace.log")


if __name__ == "__main__":
    main()
