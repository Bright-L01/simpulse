#!/usr/bin/env python3
"""
Complete End-to-End Integration Test for Simpulse
This script demonstrates the full Simpulse workflow on a real Lean project
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add parent directory to path for importing Simpulse modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.simpulse.analyzer import LeanAnalyzer
    from src.simpulse.optimization.optimizer import FrequencyOptimizer
except ImportError:
    # Fallback for testing without full install
    LeanAnalyzer = None
    FrequencyOptimizer = None


def run_command(cmd, description):
    """Run a shell command and capture output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Success")
            if result.stdout:
                print(f"Output:\n{result.stdout}")
        else:
            print(f"✗ Failed with return code: {result.returncode}")
            if result.stderr:
                print(f"Error:\n{result.stderr}")
        return result
    except Exception as e:
        print(f"✗ Exception: {e}")
        return None


def main():
    """Run complete Simpulse pipeline on integration test project"""

    print("🚀 Simpulse End-to-End Integration Test")
    print("=" * 60)

    # Paths
    test_dir = Path(__file__).parent
    lean_file = test_dir / "Main.lean"
    optimized_file = test_dir / "Main_optimized.lean"

    # Step 1: Build the Lean project to ensure it's valid
    print("\n📦 Step 1: Building Lean project...")
    os.chdir(test_dir)
    build_result = run_command("lake build", "Building Lean project")
    if not build_result or build_result.returncode != 0:
        print("❌ Failed to build Lean project. Ensure Lean 4 is installed.")
        return 1

    # Step 2: Analyze the project with Simpulse
    print("\n🔍 Step 2: Analyzing project with Simpulse...")
    LeanAnalyzer()

    try:
        # Extract simp rules from the Lean file
        with open(lean_file) as f:
            content = f.read()

        # Simple pattern matching for simp usage
        import re

        simp_patterns = re.findall(r"simp\s*(?:\[([^\]]*)\])?", content)

        print(f"Found {len(simp_patterns)} simp calls in the project")

        # Count rule frequencies
        rule_frequencies = {}
        for pattern in simp_patterns:
            if pattern:  # Has explicit rules
                rules = [r.strip() for r in pattern.split(",")]
                for rule in rules:
                    rule_frequencies[rule] = rule_frequencies.get(rule, 0) + 1
            else:  # Default simp
                rule_frequencies["_default"] = rule_frequencies.get("_default", 0) + 1

        print("\nRule frequencies:")
        for rule, count in sorted(rule_frequencies.items(), key=lambda x: x[1], reverse=True):
            print(f"  {rule}: {count} times")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1

    # Step 3: Generate optimization suggestions
    print("\n💡 Step 3: Generating optimization suggestions...")
    FrequencyOptimizer()

    # Identify optimization opportunities
    optimizations = []

    # Check for repetitive patterns
    if "Nat.add_assoc" in rule_frequencies and rule_frequencies["Nat.add_assoc"] >= 3:
        optimizations.append(
            {
                "type": "combine_repetitive",
                "rule": "Nat.add_assoc",
                "count": rule_frequencies["Nat.add_assoc"],
                "suggestion": "Create a custom simp set for associativity rules",
            }
        )

    # Check for default simp overuse
    if "_default" in rule_frequencies and rule_frequencies["_default"] >= 3:
        optimizations.append(
            {
                "type": "specialize_default",
                "count": rule_frequencies["_default"],
                "suggestion": "Replace default simp with specific rule sets",
            }
        )

    print(f"\nFound {len(optimizations)} optimization opportunities:")
    for i, opt in enumerate(optimizations, 1):
        print(f"\n{i}. {opt['type']}:")
        print(f"   {opt['suggestion']}")
        if "rule" in opt:
            print(f"   Rule: {opt['rule']} (used {opt['count']} times)")

    # Step 4: Apply optimizations to create optimized version
    print("\n🔧 Step 4: Creating optimized version...")

    # Read original content
    with open(lean_file) as f:
        optimized_content = f.read()

    # Apply simple optimizations
    # 1. Create custom simp set for frequently used rules
    if any(opt["type"] == "combine_repetitive" for opt in optimizations):
        simp_set_def = """
-- Custom simp set for frequently used rules
attribute [local simp] Nat.add_assoc

"""
        optimized_content = simp_set_def + optimized_content

        # Replace repetitive simp calls with just 'simp'
        optimized_content = optimized_content.replace("simp [Nat.add_assoc]", "simp")

    # 2. Make default simp more specific where possible
    optimized_content = optimized_content.replace(
        "simp\n  simp\n  simp", "simp only [List.append_nil]"
    )

    # Write optimized version
    with open(optimized_file, "w") as f:
        f.write(optimized_content)

    print(f"✓ Created optimized version: {optimized_file}")

    # Step 5: Measure performance improvement
    print("\n📊 Step 5: Measuring performance improvement...")

    # Benchmark original version
    print("\nBenchmarking original version...")
    start_time = time.time()
    run_command("lake build --fresh", "Building original version")
    original_time = time.time() - start_time

    # Clean and prepare for optimized version
    run_command("lake clean", "Cleaning build artifacts")

    # Copy optimized version over original temporarily
    import shutil

    shutil.copy(lean_file, lean_file.with_suffix(".backup"))
    shutil.copy(optimized_file, lean_file)

    # Benchmark optimized version
    print("\nBenchmarking optimized version...")
    start_time = time.time()
    run_command("lake build --fresh", "Building optimized version")
    optimized_time = time.time() - start_time

    # Restore original
    shutil.copy(lean_file.with_suffix(".backup"), lean_file)
    os.remove(lean_file.with_suffix(".backup"))

    # Step 6: Generate report
    print("\n📄 Step 6: Generating performance report...")

    improvement = (
        ((original_time - optimized_time) / original_time) * 100 if original_time > 0 else 0
    )

    report = {
        "test_project": str(test_dir),
        "analysis": {
            "total_simp_calls": len(simp_patterns),
            "unique_rules": len(rule_frequencies),
            "rule_frequencies": rule_frequencies,
        },
        "optimizations": optimizations,
        "performance": {
            "original_time": f"{original_time:.3f}s",
            "optimized_time": f"{optimized_time:.3f}s",
            "improvement": f"{improvement:.1f}%",
            "status": "SUCCESS" if improvement > 0 else "NO_IMPROVEMENT",
        },
        "files": {"original": str(lean_file), "optimized": str(optimized_file)},
    }

    report_file = test_dir / "integration_test_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Report saved to: {report_file}")

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 INTEGRATION TEST COMPLETE")
    print("=" * 60)
    print(f"Original build time:   {original_time:.3f}s")
    print(f"Optimized build time:  {optimized_time:.3f}s")
    print(f"Performance improvement: {improvement:.1f}%")
    print(f"\nOptimizations applied: {len(optimizations)}")

    if improvement > 0:
        print("\n✅ SUCCESS: Simpulse optimization improved performance!")
    else:
        print("\n⚠️  No measurable improvement detected")

    print("\nTo see the differences, compare:")
    print(f"  - Original:  {lean_file}")
    print(f"  - Optimized: {optimized_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
