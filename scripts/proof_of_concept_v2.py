#!/usr/bin/env python3
"""
Minimal proof that Simpulse can optimize simp rules.
Version 2: Better debugging and error handling.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SimpleMutation:
    """A simple mutation to apply to a simp rule."""
    rule_name: str
    old_annotation: str
    new_annotation: str
    description: str


def check_lean_installation() -> Tuple[bool, str]:
    """Check if Lean 4 is installed and get version info."""
    logger.info("Checking for Lean 4 installation...")
    
    try:
        # Check lake version
        lake_result = subprocess.run(
            ["lake", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        # Check lean version  
        lean_result = subprocess.run(
            ["lean", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if lake_result.returncode == 0 and lean_result.returncode == 0:
            version_info = f"Lake: {lake_result.stdout.strip()}\nLean: {lean_result.stdout.strip()}"
            logger.info(f"✓ Lean 4 found:\n{version_info}")
            return True, version_info
        else:
            logger.error("Lean commands failed")
            if lake_result.stderr:
                logger.error(f"Lake error: {lake_result.stderr}")
            if lean_result.stderr:
                logger.error(f"Lean error: {lean_result.stderr}")
            return False, "Commands failed"
            
    except FileNotFoundError:
        logger.error("Lean executables not found in PATH")
        return False, "Not found in PATH"
    except subprocess.TimeoutExpired:
        logger.error("Lean commands timed out")
        return False, "Command timeout"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False, str(e)


async def create_simple_test() -> Tuple[bool, Path]:
    """Create a minimal Lean 4 test project."""
    logger.info("Creating minimal test project...")
    
    try:
        # Create temp directory
        tmpdir = tempfile.mkdtemp(prefix="simpulse_test_")
        project_dir = Path(tmpdir) / "test_project"
        project_dir.mkdir()
        logger.info(f"Created test directory: {project_dir}")
        
        # Create minimal lakefile
        lakefile = project_dir / "lakefile.lean"
        lakefile_content = """import Lake
open Lake DSL

package simpulseTest

@[default_target]
lean_lib SimplulseTest
"""
        lakefile.write_text(lakefile_content)
        logger.debug("Created lakefile.lean")
        
        # Create test module
        test_file = project_dir / "SimplulseTest.lean"
        test_content = """-- SimplulseTest.lean
-- Minimal test for simp optimization

-- Basic arithmetic rules
@[simp] theorem my_add_zero (n : Nat) : n + 0 = n := rfl
@[simp] theorem my_zero_add (n : Nat) : 0 + n = n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [Nat.add_succ, ih]

@[simp] theorem my_mul_one (n : Nat) : n * 1 = n := by
  rw [Nat.mul_one]

-- Test theorem that uses simp
theorem test_simp : ∀ x y : Nat, (x + 0) * 1 + 0 + y = x + y := by
  intro x y
  simp [my_add_zero, my_mul_one, my_zero_add]

#check test_simp
"""
        test_file.write_text(test_content)
        logger.debug("Created test Lean file")
        
        # Initialize lake project
        logger.info("Initializing Lake project...")
        init_result = subprocess.run(
            ["lake", "update"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if init_result.returncode != 0:
            logger.error(f"Lake update failed: {init_result.stderr}")
            return False, project_dir
            
        return True, project_dir
        
    except Exception as e:
        logger.error(f"Failed to create test project: {e}")
        return False, Path("/tmp")


async def measure_build_time(project_dir: Path) -> Tuple[bool, float]:
    """Measure how long it takes to build the project."""
    logger.info("Measuring build time...")
    
    try:
        # Clean build
        subprocess.run(["lake", "clean"], cwd=project_dir, capture_output=True)
        
        # Time the build
        start_time = time.perf_counter()
        result = subprocess.run(
            ["lake", "build"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=60
        )
        end_time = time.perf_counter()
        
        build_time = end_time - start_time
        
        if result.returncode == 0:
            logger.info(f"✓ Build successful in {build_time:.3f}s")
            return True, build_time
        else:
            logger.error(f"Build failed: {result.stderr}")
            return False, 0.0
            
    except subprocess.TimeoutExpired:
        logger.error("Build timed out")
        return False, 0.0
    except Exception as e:
        logger.error(f"Build error: {e}")
        return False, 0.0


def extract_simp_rules(lean_file: Path) -> List[Tuple[str, str]]:
    """Extract simp rules from Lean file."""
    logger.info(f"Extracting simp rules from {lean_file.name}...")
    
    try:
        content = lean_file.read_text()
        rules = []
        
        # More robust pattern
        pattern = r'@\[simp\]\s+theorem\s+(\w+)'
        
        for match in re.finditer(pattern, content):
            theorem_name = match.group(1)
            rules.append(("simp", theorem_name))
            logger.debug(f"Found rule: @[simp] theorem {theorem_name}")
            
        logger.info(f"Found {len(rules)} simp rules")
        return rules
        
    except Exception as e:
        logger.error(f"Failed to extract rules: {e}")
        return []


def apply_simple_mutation(lean_file: Path, mutation: str) -> bool:
    """Apply a simple mutation to the Lean file."""
    logger.info(f"Applying mutation: {mutation}")
    
    try:
        content = lean_file.read_text()
        original_content = content
        
        if mutation == "high_priority_add_zero":
            content = content.replace(
                "@[simp] theorem my_add_zero",
                "@[simp high] theorem my_add_zero"
            )
        elif mutation == "remove_zero_add":
            content = content.replace(
                "@[simp] theorem my_zero_add",
                "theorem my_zero_add"
            )
        elif mutation == "high_priority_mul_one":
            content = content.replace(
                "@[simp] theorem my_mul_one",
                "@[simp high] theorem my_mul_one"
            )
            
        if content != original_content:
            lean_file.write_text(content)
            logger.debug("Mutation applied successfully")
            return True
        else:
            logger.warning("No changes made")
            return False
            
    except Exception as e:
        logger.error(f"Failed to apply mutation: {e}")
        return False


async def run_real_test() -> bool:
    """Run the actual proof of concept with real Lean code."""
    logger.info("\n" + "="*60)
    logger.info("SIMPULSE PROOF OF CONCEPT - REAL TEST")
    logger.info("="*60 + "\n")
    
    # Step 1: Create test project
    success, project_dir = await create_simple_test()
    if not success:
        logger.error("Failed to create test project")
        return False
        
    try:
        # Step 2: Baseline measurement
        logger.info("\n--- BASELINE MEASUREMENT ---")
        success, baseline_time = await measure_build_time(project_dir)
        if not success:
            logger.error("Failed to measure baseline")
            return False
            
        # Step 3: Extract rules
        logger.info("\n--- RULE EXTRACTION ---")
        test_file = project_dir / "SimplulseTest.lean"
        rules = extract_simp_rules(test_file)
        if not rules:
            logger.error("No rules found")
            return False
            
        # Step 4: Test mutations
        logger.info("\n--- TESTING MUTATIONS ---")
        mutations = [
            ("high_priority_add_zero", "Give high priority to my_add_zero"),
            ("high_priority_mul_one", "Give high priority to my_mul_one"),
            ("remove_zero_add", "Remove simp from my_zero_add")
        ]
        
        results = []
        for mut_id, mut_desc in mutations:
            logger.info(f"\nTesting: {mut_desc}")
            
            # Apply mutation
            if apply_simple_mutation(test_file, mut_id):
                # Measure new time
                success, new_time = await measure_build_time(project_dir)
                if success:
                    improvement = ((baseline_time - new_time) / baseline_time) * 100
                    results.append((mut_desc, baseline_time, new_time, improvement))
                    logger.info(f"Result: {new_time:.3f}s ({improvement:+.1f}%)")
                else:
                    logger.warning("Build failed with mutation")
                    
                # Revert for next test
                success, _ = await create_simple_test()  # Recreate clean state
                if not success:
                    break
        
        # Step 5: Report results
        logger.info("\n" + "="*60)
        logger.info("RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"Baseline build time: {baseline_time:.3f}s")
        logger.info("\nMutation results:")
        
        best_improvement = 0
        for desc, base, new, imp in results:
            logger.info(f"  {desc}: {new:.3f}s ({imp:+.1f}%)")
            if imp > best_improvement:
                best_improvement = imp
                
        if best_improvement > 0:
            logger.info(f"\n✅ PROOF OF CONCEPT SUCCESSFUL!")
            logger.info(f"Best improvement: {best_improvement:.1f}%")
            return True
        else:
            logger.info(f"\n⚠️ No improvements found")
            logger.info("This might be due to:")
            logger.info("  - Test case too simple")
            logger.info("  - Build time dominated by other factors")
            logger.info("  - Need more sophisticated mutations")
            return False
            
    finally:
        # Cleanup
        if project_dir.exists():
            import shutil
            shutil.rmtree(project_dir.parent)
            logger.debug(f"Cleaned up {project_dir.parent}")


async def main():
    """Main entry point."""
    # Check Lean installation
    installed, version_info = check_lean_installation()
    
    if not installed:
        logger.error("\n❌ Lean 4 is not installed!")
        logger.info("\nPlease run: ./run_proof_of_concept.sh")
        logger.info("Or install manually:")
        logger.info("  curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh")
        return False
        
    # Run real test
    return await run_real_test()


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)