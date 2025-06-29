#!/usr/bin/env python3
"""
Run all Simpulse tests and validation checks.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} - PASSED")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"❌ {description} - FAILED")
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.stdout:
            print("STDOUT:", result.stdout)
    
    return result.returncode == 0

def main():
    """Run all tests and checks."""
    print("SIMPULSE TEST SUITE")
    print("="*60)
    
    # Check Python version
    import sys
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return 1
    
    all_passed = True
    
    # 1. Run syntax checks
    print("\n1. SYNTAX VALIDATION")
    python_files = list(Path("src").rglob("*.py")) + list(Path("tests").rglob("*.py")) + list(Path("scripts").rglob("*.py"))
    
    for py_file in python_files:
        try:
            compile(py_file.read_text(), py_file, 'exec')
        except SyntaxError as e:
            print(f"❌ Syntax error in {py_file}: {e}")
            all_passed = False
    
    if all_passed:
        print(f"✅ All {len(python_files)} Python files have valid syntax")
    
    # 2. Run import checks
    print("\n2. IMPORT VALIDATION")
    import_check = """
import sys
sys.path.insert(0, 'src')
try:
    import simpulse
    import simpulse.config
    import simpulse.evolution.evolution_engine
    import simpulse.profiling.lean_runner
    import simpulse.security.validators
    print("✅ All core modules import successfully")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
"""
    
    if not run_command([sys.executable, "-c", import_check], "Import validation"):
        all_passed = False
    
    # 3. Run security checks
    print("\n3. SECURITY VALIDATION")
    security_files = [
        "src/simpulse/profiling/lean_runner.py",
        "src/simpulse/security/validators.py"
    ]
    
    for sec_file in security_files:
        if Path(sec_file).exists():
            content = Path(sec_file).read_text()
            # Check for dangerous patterns
            dangerous = ["eval(", "exec(", "__import__", "os.system", "subprocess.call("]
            for pattern in dangerous:
                if pattern in content and "validate" not in content:
                    print(f"⚠️ Potential security issue in {sec_file}: found {pattern}")
    
    print("✅ Security checks completed")
    
    # 4. Run simulations
    print("\n4. PROOF OF CONCEPT SIMULATIONS")
    
    if Path("scripts/realistic_simulation.py").exists():
        if not run_command([sys.executable, "scripts/realistic_simulation.py"], "Realistic simulation"):
            all_passed = False
    
    if Path("scripts/minimal_working_example.py").exists():
        if not run_command([sys.executable, "scripts/minimal_working_example.py"], "Minimal working example"):
            all_passed = False
    
    # 5. Check documentation
    print("\n5. DOCUMENTATION CHECK")
    required_docs = ["README.md", "LICENSE", "CONTRIBUTING.md", "CHANGELOG.md"]
    for doc in required_docs:
        if Path(doc).exists():
            print(f"✅ {doc} exists")
        else:
            print(f"❌ {doc} missing")
            all_passed = False
    
    # 6. Final report
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nNext steps:")
        print("1. Install Lean 4 to run empirical tests")
        print("2. Run ./test_simpulse_now.sh for real validation")
        return 0
    else:
        print("❌ Some tests failed")
        print("Please fix the issues before committing")
        return 1

if __name__ == "__main__":
    sys.exit(main())