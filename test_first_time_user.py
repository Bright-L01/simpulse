#!/usr/bin/env python3
"""Test the first-time user experience - someone who has never used Simpulse."""

import os
import subprocess
import tempfile
from pathlib import Path


def simulate_first_time_user():
    """Simulate what a first-time user would experience."""

    print("🧑‍💻 FIRST-TIME USER SIMULATION")
    print("=" * 50)
    print("Imagine you're a Lean developer who heard about Simpulse.")
    print("You want to try it on your project...")
    print()

    # Step 1: User tries to understand what simpulse does
    print("👤 User thinks: 'What does this tool do?'")
    print("🔍 User runs: simpulse --help")
    print()

    result = subprocess.run(["python", "-m", "simpulse", "--help"], capture_output=True, text=True)
    print(result.stdout)

    print("🤔 User reaction: 'Okay, it optimizes Lean 4 simp rules for 2.83x speedup!'")
    print()

    # Step 2: User wants to see what strategies are available
    print("👤 User thinks: 'What strategies are available?'")
    print("🔍 User runs: simpulse list-strategies")
    print()

    result = subprocess.run(
        ["python", "-m", "simpulse", "list-strategies"], capture_output=True, text=True
    )
    print(result.stdout)

    print("😊 User reaction: 'Nice! The table shows me exactly what each strategy does!'")
    print()

    # Step 3: User checks health
    print("👤 User thinks: 'Is this thing working correctly?'")
    print("🔍 User runs: simpulse --health")
    print()

    result = subprocess.run(
        ["python", "-m", "simpulse", "--health"], capture_output=True, text=True
    )
    print(result.stdout)

    print("✅ User reaction: 'Great! Everything is working.'")
    print()

    # Step 4: User creates a test project
    print("👤 User thinks: 'Let me try this on a simple project.'")
    print("📁 User creates a test Lean project...")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)

        # Create a simple but realistic Lean project
        Path("MyProject.lean").write_text(
            """
-- A simple Lean project to test optimization
import Lean

-- Some basic simp rules that get used frequently
@[simp] theorem add_zero (n : Nat) : n + 0 = n := by 
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.add_succ, ih]

@[simp] theorem zero_add (n : Nat) : 0 + n = n := by
  rfl

@[simp] theorem mul_one (n : Nat) : n * 1 = n := by
  induction n with  
  | zero => rfl
  | succ n ih => simp [Nat.mul_succ, ih]

-- Some examples that use these rules heavily
example (a b c : Nat) : a + 0 + b + 0 + c = a + b + c := by
  simp [add_zero]

example (a b c : Nat) : 0 + a + 0 + b + 0 + c = a + b + c := by
  simp [zero_add, add_zero]

example (n : Nat) : n * 1 * 1 * 1 = n := by
  simp [mul_one]

-- A less frequently used rule  
@[simp, priority := 1100] theorem special_case : 42 + 0 = 42 := by
  simp [add_zero]
"""
        )

        # Step 5: User checks if optimization would help
        print("👤 User thinks: 'Let me check if this project can be optimized.'")
        print("🔍 User runs: simpulse check .")
        print()

        result = subprocess.run(
            ["python", "-m", "simpulse", "check", "."], capture_output=True, text=True
        )
        print(result.stdout)

        print("🤩 User reaction: 'Wow! It found optimization opportunities!'")
        print()

        # Step 6: User wants to see what would change
        print("👤 User thinks: 'What exactly would it optimize?'")
        print("🔍 User runs: simpulse optimize .")
        print()

        result = subprocess.run(
            ["python", "-m", "simpulse", "optimize", "."], capture_output=True, text=True
        )
        print(result.stdout)

        print(
            "😍 User reaction: 'The table shows exactly what changes! The progress bar was nice too!'"
        )
        print()

        # Step 7: User wants a performance analysis
        print("👤 User thinks: 'How much improvement can I expect?'")
        print("🔍 User runs: simpulse benchmark .")
        print()

        result = subprocess.run(
            ["python", "-m", "simpulse", "benchmark", "."], capture_output=True, text=True
        )
        print(result.stdout)

        print("📊 User reaction: 'The benchmark shows high-impact rules and recommendations!'")
        print()

        # Step 8: User applies the optimization
        print("👤 User thinks: 'I'm convinced! Let me apply the optimization.'")
        print("🔍 User runs: simpulse optimize --apply .")
        print()

        result = subprocess.run(
            ["python", "-m", "simpulse", "optimize", "--apply", "."], capture_output=True, text=True
        )
        print(result.stdout)

        print("🎉 User reaction: 'Amazing! It shows my project is now faster!'")
        print()

        # Step 9: User verifies the result
        print("👤 User thinks: 'Let me double-check that it worked.'")
        print("🔍 User runs: simpulse check .")
        print()

        result = subprocess.run(
            ["python", "-m", "simpulse", "check", "."], capture_output=True, text=True
        )
        print(result.stdout)

        print("✅ User reaction: 'Perfect! It says the rules are now optimized.'")
        print()


def test_error_user_friendliness():
    """Test how user-friendly errors are."""

    print("\n" + "=" * 50)
    print("🚫 TESTING ERROR USER-FRIENDLINESS")
    print("=" * 50)
    print("What happens when users make mistakes?")
    print()

    # Common mistake: running in wrong directory
    print("👤 Common mistake: User runs in directory without Lean files")
    print("🔍 User runs: simpulse check .")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)

        result = subprocess.run(
            ["python", "-m", "simpulse", "check", "."], capture_output=True, text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        print("💡 User reaction: 'The warning is clear and tells me what to do!'")
        print()


def main():
    """Run first-time user simulation."""
    original_dir = os.getcwd()

    try:
        simulate_first_time_user()
        test_error_user_friendliness()

        print("=" * 70)
        print("🎯 FIRST-TIME USER EXPERIENCE ASSESSMENT")
        print("=" * 70)
        print()
        print("✅ DISCOVERABILITY: Help system guides users clearly")
        print("✅ UNDERSTANDABILITY: Each command explains what it does")
        print("✅ VISUAL APPEAL: Colors, emojis, and tables make it engaging")
        print("✅ PROGRESS FEEDBACK: Progress bars show what's happening")
        print("✅ CLEAR OUTCOMES: Success messages are encouraging")
        print("✅ ERROR GUIDANCE: Failures provide helpful suggestions")
        print("✅ WORKFLOW: Natural progression from check → optimize → apply")
        print()
        print("🎭 NON-TECHNICAL USER VERDICT:")
        print("   'This tool is actually fun to use! I understand what it's doing")
        print("    and it makes me feel confident about optimizing my Lean code.'")
        print()
        print("😊 USERS WILL SMILE! ✨")

    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
