"""Example usage of Simpulse profiling tools.

This script demonstrates how to use LeanRunner and TraceParser
to profile Lean 4 code and analyze simp tactic performance.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simpulse.profiling import LeanRunner, TraceParser


async def profile_lean_file(file_path: Path):
    """Profile a single Lean file."""
    print(f"Profiling {file_path}...")
    
    # Create runner
    runner = LeanRunner()
    
    # Run with profiling enabled
    result = await runner.run_lean(
        file_path=file_path,
        trace_flags=["profiler.output", "Meta.Tactic.simp"],
        timeout=60.0
    )
    
    print(f"Execution completed with return code: {result.returncode}")
    print(f"Elapsed time: {result.elapsed_time:.2f} seconds")
    
    if result.stderr:
        print("\nStderr output:")
        print(result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr)
        
    # Parse trace output
    parser = TraceParser()
    report = parser.parse_content(result.stderr)
    
    print(f"\nFound {len(report.entries)} profile entries")
    print(f"Total profiled time: {report.total_time_ms:.2f} ms")
    
    # Show top entries
    print("\nTop 5 entries by time:")
    for i, entry in enumerate(report.get_top_entries(5), 1):
        print(f"  {i}. {entry.name}: {entry.elapsed_ms:.2f} ms ({entry.count} calls)")


async def profile_module(module_name: str):
    """Profile a Lean module by name."""
    print(f"Profiling module: {module_name}")
    
    runner = LeanRunner()
    
    # Profile the module
    result, profile_data = await runner.profile_module(
        module_name=module_name,
        options={
            'trace_flags': ['profiler.output', 'Meta.Tactic.simp'],
            'timeout': 120.0,
            'parse_output': True
        }
    )
    
    if result.success:
        print(f"✓ Module profiled successfully in {result.elapsed_time:.2f} seconds")
    else:
        print(f"✗ Profiling failed with return code {result.returncode}")
        if result.stderr:
            print(f"Error: {result.stderr[:200]}...")
        return
        
    # Parse and analyze simp statistics
    parser = TraceParser()
    simp_stats = parser.parse_simp_trace(result.stderr)
    
    print(f"\nSimp Statistics:")
    print(f"  Total rewrites: {simp_stats['total_rewrites']}")
    print(f"  Successful: {simp_stats['successful_rewrites']}")
    print(f"  Failed: {simp_stats['failed_rewrites']}")
    print(f"  Unique theorems used: {len(simp_stats['unique_theorems'])}")
    
    if simp_stats['most_used_theorems']:
        print("\n  Most used theorems:")
        for theorem, count in list(simp_stats['most_used_theorems'].items())[:5]:
            print(f"    - {theorem}: {count} times")


async def main():
    """Main example function."""
    print("Simpulse Profiling Example")
    print("=" * 50)
    
    # Example 1: Profile a specific file
    test_file = Path("test.lean")
    if test_file.exists():
        await profile_lean_file(test_file)
    else:
        print(f"Skipping file profiling - {test_file} not found")
    
    print("\n" + "=" * 50 + "\n")
    
    # Example 2: Profile a module (requires a Lean project with Mathlib)
    # Uncomment to test with a real Lean project:
    # await profile_module("Mathlib.Data.List.Basic")
    
    # Example 3: Get simp diagnostics
    if test_file.exists():
        print("Getting simp diagnostics...")
        runner = LeanRunner()
        result, diagnostics = await runner.get_simp_diagnostics(test_file)
        
        if diagnostics:
            print(f"Found {len(diagnostics['rewrites'])} rewrite traces")
            print(f"Found {len(diagnostics['failures'])} failure traces")


if __name__ == "__main__":
    asyncio.run(main())