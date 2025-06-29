#!/usr/bin/env python3
"""
Radical cleanup: Delete 40% of the codebase.
Keep only what's proven useful.
"""

import shutil
from pathlib import Path


class RadicalCleanup:
    """Remove all complexity that isn't proven useful."""

    def __init__(self):
        self.repo_root = Path(__file__).parent.parent
        self.src_path = self.repo_root / "src" / "simpulse"
        self.deleted_count = 0
        self.kept_count = 0

    def execute_cleanup(self):
        """Remove all unused code ruthlessly."""

        print("=" * 70)
        print("RADICAL CODEBASE CLEANUP")
        print("=" * 70)
        print("Removing all unproven complexity...\n")

        # Step 1: Delete entire unnecessary directories
        unnecessary_dirs = [
            self.src_path / "strategies",  # Premature optimization
            self.src_path / "web",  # No users yet
            self.src_path / "marketing",  # LOL
            self.src_path / "benchmarks",  # Use real measurements
            self.src_path / "deployment",  # Overengineered
            self.src_path / "integrations",  # Not proven useful
            self.src_path / "monitoring",  # Premature
            self.src_path / "core" / "documentation.py",  # Auto-generated docs
            self.repo_root / "marketing",  # Definitely not needed
            self.repo_root / "docker",  # Too early
        ]

        print("1. DELETING UNNECESSARY DIRECTORIES")
        print("-" * 50)
        for dir_path in unnecessary_dirs:
            if dir_path.exists():
                if dir_path.is_dir():
                    shutil.rmtree(dir_path)
                else:
                    dir_path.unlink()
                print(f"   ❌ Deleted: {dir_path.relative_to(self.repo_root)}")
                self.deleted_count += 1

        # Step 2: Delete unnecessary files
        unnecessary_files = [
            self.src_path / "cli_v2.py",  # Keep only simple CLI
            self.src_path / "config.py",  # Overengineered
            self.src_path / "analysis" / "complexity_analyzer.py",  # Not useful
            self.src_path / "analysis" / "dependency_analyzer.py",  # Not useful
            self.src_path / "analysis" / "performance_predictor.py",  # Doesn't work
            self.src_path / "claude" / "prompt_builder.py",  # Too complex
            self.src_path / "evolution" / "models_v2.py",  # Duplicate
            self.src_path / "evolution" / "workspace_manager.py",  # Overengineered
            self.src_path / "reporting" / "html_dashboard.py",  # Premature
            self.src_path / "reporting" / "github_commenter.py",  # Not needed yet
            self.repo_root / ".github" / "workflows" / "deploy.yml",  # No deployment
            self.repo_root / ".github" / "workflows" / "publish.yml",  # No package
        ]

        print("\n2. DELETING UNNECESSARY FILES")
        print("-" * 50)
        for file_path in unnecessary_files:
            if file_path.exists():
                file_path.unlink()
                print(f"   ❌ Deleted: {file_path.relative_to(self.repo_root)}")
                self.deleted_count += 1

        # Step 3: Simplify remaining modules
        print("\n3. SIMPLIFYING REMAINING MODULES")
        print("-" * 50)
        self.simplify_evolution_engine()
        self.simplify_cli()
        self.create_minimal_config()

        # Step 4: Delete test files for deleted modules
        print("\n4. CLEANING UP TESTS")
        print("-" * 50)
        self.cleanup_tests()

        # Step 5: Update imports and dependencies
        print("\n5. UPDATING DEPENDENCIES")
        print("-" * 50)
        self.update_dependencies()

        # Summary
        print("\n" + "=" * 70)
        print("CLEANUP COMPLETE")
        print("=" * 70)
        print(f"Files deleted: {self.deleted_count}")
        print(f"Files kept: {self.kept_count}")

        # Count remaining files
        remaining = list(self.src_path.rglob("*.py"))
        print(f"\nRemaining Python files: {len(remaining)}")
        print(
            "Target was <30 files. "
            + ("✅ SUCCESS!" if len(remaining) < 30 else "❌ Still too many")
        )

        return len(remaining) < 30

    def simplify_evolution_engine(self):
        """Strip evolution engine to bare minimum."""
        engine_path = self.src_path / "evolution" / "evolution_engine.py"
        if not engine_path.exists():
            return

        print(f"   📝 Simplifying {engine_path.name}")

        # Read current content
        engine_path.read_text()

        # Create minimal version
        minimal_engine = '''"""Minimal evolution engine for simp optimization."""

import asyncio
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from .rule_extractor import RuleExtractor
from .mutation_applicator import MutationApplicator
from ..profiling.lean_runner import LeanRunner


@dataclass
class OptimizationResult:
    """Result of optimization attempt."""
    improved: bool
    improvement_percent: float = 0.0
    best_mutation: Optional[str] = None
    baseline_time: float = 0.0
    optimized_time: float = 0.0


class SimpleEvolutionEngine:
    """Minimal engine that just tries priority swaps."""
    
    def __init__(self):
        self.extractor = RuleExtractor()
        self.applicator = MutationApplicator()
        self.runner = LeanRunner()
        
    async def optimize_file(self, lean_file: Path) -> OptimizationResult:
        """Try simple priority optimizations."""
        
        # Extract rules
        module_rules = self.extractor.extract_rules_from_file(lean_file)
        rules = module_rules.rules
        
        if len(rules) < 2:
            return OptimizationResult(improved=False)
            
        # Profile baseline
        baseline = await self.runner.profile_file(lean_file)
        baseline_time = baseline.get("total_time", 0)
        
        # Try swapping priorities of rule pairs
        best_time = baseline_time
        best_mutation = None
        
        for i in range(min(len(rules), 5)):
            for j in range(i + 1, min(len(rules), 5)):
                # Create swap mutation
                mutation = f"Swap priorities: {rules[i].name} <-> {rules[j].name}"
                
                # Apply and test
                try:
                    mutated_file = self.applicator.apply_simple_swap(
                        lean_file, rules[i], rules[j]
                    )
                    
                    result = await self.runner.profile_file(mutated_file)
                    time = result.get("total_time", float('inf'))
                    
                    if time < best_time:
                        best_time = time
                        best_mutation = mutation
                        
                except Exception:
                    continue
                    
        # Calculate improvement
        if best_mutation and best_time < baseline_time:
            improvement = (baseline_time - best_time) / baseline_time * 100
            return OptimizationResult(
                improved=True,
                improvement_percent=improvement,
                best_mutation=best_mutation,
                baseline_time=baseline_time,
                optimized_time=best_time
            )
        else:
            return OptimizationResult(
                improved=False,
                baseline_time=baseline_time,
                optimized_time=baseline_time
            )
'''

        engine_path.write_text(minimal_engine)
        self.kept_count += 1

    def simplify_cli(self):
        """Create a minimal CLI."""
        cli_path = self.src_path / "cli.py"

        print(f"   📝 Creating minimal {cli_path.name}")

        minimal_cli = '''#!/usr/bin/env python3
"""Minimal Simpulse CLI - just the essentials."""

import asyncio
import sys
from pathlib import Path

from .evolution.evolution_engine import SimpleEvolutionEngine


async def optimize_command(file_path: str):
    """Optimize a single Lean file."""
    
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File {file_path} not found")
        return 1
        
    print(f"Optimizing {path.name}...")
    
    engine = SimpleEvolutionEngine()
    result = await engine.optimize_file(path)
    
    if result.improved:
        print(f"✅ Success! {result.improvement_percent:.1f}% improvement")
        print(f"   Baseline: {result.baseline_time:.2f}ms")
        print(f"   Optimized: {result.optimized_time:.2f}ms")
        print(f"   Best mutation: {result.best_mutation}")
    else:
        print("❌ No improvement found")
        print("   Try a file with more simp rules")
        
    return 0 if result.improved else 1


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: simpulse optimize <file.lean>")
        print("       simpulse --help")
        return 1
        
    command = sys.argv[1]
    
    if command == "--help" or command == "-h":
        print("Simpulse - Lean 4 simp optimizer")
        print()
        print("Commands:")
        print("  optimize <file>  Optimize simp rules in a Lean file")
        print("  --help          Show this help")
        return 0
        
    elif command == "optimize" and len(sys.argv) > 2:
        return asyncio.run(optimize_command(sys.argv[2]))
        
    else:
        print(f"Unknown command: {command}")
        print("Run 'simpulse --help' for usage")
        return 1


if __name__ == "__main__":
    sys.exit(main())
'''

        cli_path.write_text(minimal_cli)
        self.kept_count += 1

    def create_minimal_config(self):
        """Create minimal configuration."""
        init_path = self.src_path / "__init__.py"

        print("   📝 Creating minimal __init__.py")

        minimal_init = '''"""Simpulse - Minimal simp optimizer for Lean 4."""

__version__ = "0.1.0-alpha"
__author__ = "Simpulse Contributors"

# That's it. No complex configuration needed.
'''

        init_path.write_text(minimal_init)
        self.kept_count += 1

    def cleanup_tests(self):
        """Remove tests for deleted modules."""
        test_dir = self.repo_root / "tests"
        if not test_dir.exists():
            return

        # Tests to remove
        tests_to_remove = [
            "test_strategies.py",
            "test_web.py",
            "test_deployment.py",
            "test_monitoring.py",
            "test_benchmarks.py",
            "test_config.py",
            "test_cli_v2.py",
        ]

        for test_file in tests_to_remove:
            test_path = test_dir / test_file
            if test_path.exists():
                test_path.unlink()
                print(f"   ❌ Deleted test: {test_file}")
                self.deleted_count += 1

    def update_dependencies(self):
        """Update pyproject.toml to remove unnecessary dependencies."""
        pyproject_path = self.repo_root / "pyproject.toml"
        if not pyproject_path.exists():
            return

        print("   📝 Updating pyproject.toml")

        # Read current content
        content = pyproject_path.read_text()

        # Remove web/deployment dependencies
        lines = content.split("\n")
        filtered_lines = []
        skip_next = False

        for line in lines:
            if any(
                dep in line for dep in ["aiohttp", "jinja2", "prometheus", "docker"]
            ):
                skip_next = True
                continue
            if skip_next and line.strip() == "":
                skip_next = False
                continue
            filtered_lines.append(line)

        pyproject_path.write_text("\n".join(filtered_lines))

        # Also update requirements.txt if it exists
        req_path = self.repo_root / "requirements.txt"
        if req_path.exists():
            print("   📝 Clearing requirements.txt")
            req_path.write_text(
                "# No external dependencies - using only Python stdlib\n"
            )


def main():
    """Execute radical cleanup."""
    print("⚠️  WARNING: This will DELETE 40% of the codebase!")
    print("⚠️  Make sure you have committed all changes first!")
    print()

    response = input("Continue? (yes/no): ")
    if response.lower() != "yes":
        print("Cleanup cancelled.")
        return

    cleaner = RadicalCleanup()
    success = cleaner.execute_cleanup()

    if success:
        print("\n✅ Cleanup successful! Codebase is now minimal and focused.")
        print("   Next step: Test on real projects")
    else:
        print("\n⚠️  Cleanup complete but still have too many files.")
        print("   Consider more aggressive removal")


if __name__ == "__main__":
    main()
