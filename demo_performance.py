#!/usr/bin/env python3
"""Quick demonstration of Simpulse performance improvements."""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from simpulse.optimization.fast_optimizer import FastOptimizer
from simpulse.optimization.optimizer import SimpOptimizer


def create_demo_project():
    """Create a small demo project."""
    test_dir = Path(tempfile.mkdtemp(prefix="simpulse_demo_"))

    # Create 5 test files
    for i in range(5):
        content = f"""
namespace Demo{i}

@[simp] theorem demo_basic_{i}_1 : (0 + {i}) = {i} := by simp
@[simp] theorem demo_basic_{i}_2 : ({i} + 0) = {i} := by simp
@[simp] theorem demo_list_{i} : List.length [1, 2, 3] = 3 := by simp
@[simp, priority := 200] theorem demo_prio_{i} : {i} * 1 = {i} := by simp
@[simp] theorem demo_mul_{i} : ({i} * 0) = 0 := by simp

end Demo{i}
"""
        (test_dir / f"Demo{i}.lean").write_text(content)

    return test_dir


def main():
    print("Simpulse Performance Demo")
    print("=" * 50)

    # Create demo project
    project_path = create_demo_project()
    print(f"\nCreated demo project with 5 files at: {project_path}")

    # List files in project
    lean_files = list(project_path.glob("**/*.lean"))
    print(f"Found {len(lean_files)} Lean files")
    for f in lean_files[:3]:
        print(f"  - {f.name}")

    # Test original implementation
    print("\n1. Original Implementation:")
    start = time.time()
    optimizer1 = SimpOptimizer()
    analysis1 = optimizer1.analyze(project_path)
    opt1 = optimizer1.optimize(analysis1)
    time1 = time.time() - start

    print(f"   - Time: {time1:.3f}s")
    print(f"   - Rules found: {len(analysis1['rules'])}")
    print(f"   - Optimizations: {opt1.rules_changed}")

    # Test optimized implementation
    print("\n2. Optimized Implementation:")
    start = time.time()
    optimizer2 = FastOptimizer()
    analysis2 = optimizer2.analyze(project_path, use_parallel=True)
    opt2 = optimizer2.optimize(analysis2)
    time2 = time.time() - start

    print(f"   - Time: {time2:.3f}s")
    print(f"   - Rules found: {len(analysis2['rules'])}")
    print(f"   - Optimizations: {opt2['rules_changed']}")

    # Show improvement
    speedup = time1 / time2 if time2 > 0 else 1
    print(f"\n✨ Performance Improvement: {speedup:.2f}x faster!")

    # Show cache efficiency
    stats = optimizer2.extractor.get_statistics()
    print(f"\n3. Cache Statistics:")
    print(f"   - Cache hits: {stats['cache_hits']}")
    print(f"   - Cache misses: {stats['cache_misses']}")
    print(f"   - Hit rate: {stats['cache_hit_rate']:.1%}")

    # Cleanup
    import shutil

    shutil.rmtree(project_path)

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
