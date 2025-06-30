#!/usr/bin/env python3
"""Create a realistic test Lean project with many simp rules for benchmarking."""

import os
import sys
from pathlib import Path

def create_test_project(project_name="test_simp_optimization"):
    """Create a test Lean project with many simp rules."""
    project_dir = Path(project_name)
    if project_dir.exists():
        print(f"⚠️  {project_dir} already exists. Removing...")
        import shutil
        shutil.rmtree(project_dir)
    
    # Create project
    os.system(f"lake new {project_name}")
    
    # Create multiple modules with simp rules
    src_dir = project_dir / f"{project_name.replace('_', ' ').title().replace(' ', '')}"
    
    # Module 1: Basic arithmetic with many simple rules (should be high priority)
    basic_content = """-- Basic arithmetic simplifications
namespace Basic

"""
    # Add 30 simple arithmetic rules
    for i in range(30):
        basic_content += f"""@[simp] theorem add_zero_{i} (n : Nat) : n + 0 = n := Nat.add_zero n
@[simp] theorem zero_add_{i} (n : Nat) : 0 + n = n := Nat.zero_add n
@[simp] theorem mul_one_{i} (n : Nat) : n * 1 = n := Nat.mul_one n
"""
    
    basic_content += "\nend Basic\n"
    (src_dir / "Basic.lean").write_text(basic_content)
    
    # Module 2: Complex rules (should be low priority)
    complex_content = """-- Complex pattern matching rules
namespace Complex

"""
    # Add 20 complex rules
    for i in range(20):
        complex_content += f"""@[simp] theorem complex_pattern_{i} (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_{i} (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
"""
    
    complex_content += "\nend Complex\n"
    (src_dir / "Complex.lean").write_text(complex_content)
    
    # Module 3: Test cases that use simp heavily
    test_content = """-- Test module that uses simp extensively
import TestSimpOptimization.Basic
import TestSimpOptimization.Complex

namespace Tests

-- Generate 100 test theorems that use simp
"""
    for i in range(100):
        test_content += f"""theorem test_{i} (a b c : Nat) : 
  (a + 0) * 1 + (0 + b) * (c * 1) = a + b * c := by simp [Basic.add_zero_0, Basic.mul_one_0]

"""
    
    test_content += """
-- Large test that uses many simp rules
theorem large_test (n : Nat) : 
  (n + 0) * 1 + 0 * n + n * 1 + 0 + n = 3 * n := by
  simp [Basic.add_zero_0, Basic.zero_add_0, Basic.mul_one_0]
  ring

end Tests
"""
    (src_dir / "Tests.lean").write_text(test_content)
    
    # Update the main module to import all
    main_content = f"""import {project_name.replace('_', ' ').title().replace(' ', '')}.Basic
import {project_name.replace('_', ' ').title().replace(' ', '')}.Complex  
import {project_name.replace('_', ' ').title().replace(' ', '')}.Tests

def main : IO Unit :=
  IO.println "Test project for simp optimization"
"""
    (src_dir / ".." / f"{project_name.replace('_', ' ').title().replace(' ', '')}.lean").write_text(main_content)
    
    print(f"✅ Created test project: {project_dir}")
    print(f"   - 90 simp rules (60 simple, 30 complex)")
    print(f"   - 100 test theorems using simp")
    return project_dir

if __name__ == "__main__":
    project_name = sys.argv[1] if len(sys.argv) > 1 else "test_simp_optimization"
    create_test_project(project_name)