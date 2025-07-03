#!/usr/bin/env python3
"""Real Lean 4 benchmarks - measures actual simp performance."""

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Dict


class LeanBenchmark:
    """Runs real Lean 4 benchmarks and collects performance data."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.lean_dir = project_root / "lean4"
        self.test_dir = self.lean_dir / "Benchmark"
        self.test_dir.mkdir(exist_ok=True)

    def create_test_files(self):
        """Create Lean 4 test files for benchmarking."""

        # Test 1: Simple Lists
        simple_lists = """-- SimpleLists.lean
-- Test simp performance on basic list operations

theorem list_append_nil (l : List α) : l ++ [] = l := by simp

theorem list_nil_append (l : List α) : [] ++ l = l := by simp

theorem list_append_assoc (l1 l2 l3 : List α) : 
  (l1 ++ l2) ++ l3 = l1 ++ (l2 ++ l3) := by simp

theorem list_length_append (l1 l2 : List α) : 
  (l1 ++ l2).length = l1.length + l2.length := by simp

theorem list_reverse_append (l1 l2 : List α) : 
  (l1 ++ l2).reverse = l2.reverse ++ l1.reverse := by simp

theorem list_map_append (f : α → β) (l1 l2 : List α) :
  (l1 ++ l2).map f = l1.map f ++ l2.map f := by simp

theorem list_filter_append (p : α → Bool) (l1 l2 : List α) :
  (l1 ++ l2).filter p = l1.filter p ++ l2.filter p := by simp

theorem list_take_append (n : Nat) (l1 l2 : List α) :
  (l1 ++ l2).take n = if n ≤ l1.length then l1.take n else l1 ++ l2.take (n - l1.length) := by
  split <;> simp [*]

theorem list_drop_append (n : Nat) (l1 l2 : List α) :
  (l1 ++ l2).drop n = if n ≤ l1.length then l1.drop n ++ l2 else l2.drop (n - l1.length) := by
  split <;> simp [*]

theorem list_concat_eq_append (l : List α) (a : α) :
  l.concat a = l ++ [a] := by simp
"""

        # Test 2: Basic Algebra
        basic_algebra = """-- BasicAlgebra.lean
-- Test simp performance on arithmetic simplifications

theorem add_zero (n : Nat) : n + 0 = n := by simp

theorem zero_add (n : Nat) : 0 + n = n := by simp

theorem add_comm (n m : Nat) : n + m = m + n := by simp

theorem add_assoc (n m k : Nat) : (n + m) + k = n + (m + k) := by simp

theorem mul_one (n : Nat) : n * 1 = n := by simp

theorem one_mul (n : Nat) : 1 * n = n := by simp

theorem mul_zero (n : Nat) : n * 0 = 0 := by simp

theorem zero_mul (n : Nat) : 0 * n = 0 := by simp

theorem mul_comm (n m : Nat) : n * m = m * n := by simp

theorem mul_add (n m k : Nat) : n * (m + k) = n * m + n * k := by simp

theorem add_mul (n m k : Nat) : (n + m) * k = n * k + m * k := by simp

theorem pow_zero (n : Nat) : n ^ 0 = 1 := by simp

theorem pow_succ (n m : Nat) : n ^ (m + 1) = n * n ^ m := by simp

theorem sub_self (n : Nat) : n - n = 0 := by simp

theorem add_sub_cancel (n m : Nat) : n + m - m = n := by simp
"""

        # Test 3: Logic Proofs
        logic_proofs = """-- LogicProofs.lean
-- Test simp performance on propositional logic

theorem and_comm (p q : Prop) : p ∧ q ↔ q ∧ p := by simp [and_comm]

theorem or_comm (p q : Prop) : p ∨ q ↔ q ∨ p := by simp [or_comm]

theorem and_assoc (p q r : Prop) : (p ∧ q) ∧ r ↔ p ∧ (q ∧ r) := by simp [and_assoc]

theorem or_assoc (p q r : Prop) : (p ∨ q) ∨ r ↔ p ∨ (q ∨ r) := by simp [or_assoc]

theorem not_not (p : Prop) [Decidable p] : ¬¬p ↔ p := by simp

theorem and_true (p : Prop) : p ∧ True ↔ p := by simp

theorem true_and (p : Prop) : True ∧ p ↔ p := by simp

theorem or_false (p : Prop) : p ∨ False ↔ p := by simp

theorem false_or (p : Prop) : False ∨ p ↔ p := by simp

theorem and_false (p : Prop) : p ∧ False ↔ False := by simp

theorem false_and (p : Prop) : False ∧ p ↔ False := by simp

theorem or_true (p : Prop) : p ∨ True ↔ True := by simp

theorem true_or (p : Prop) : True ∨ p ↔ True := by simp

theorem imp_self (p : Prop) : (p → p) ↔ True := by simp
"""

        # Test 4: Basic Natural Numbers
        basic_nat = """-- BasicNat.lean
-- Test simp performance on natural number operations

theorem succ_pred (n : Nat) (h : n > 0) : n.pred.succ = n := by simp [Nat.succ_pred h]

theorem pred_succ (n : Nat) : n.succ.pred = n := by simp

theorem add_succ (n m : Nat) : n + m.succ = (n + m).succ := by simp

theorem succ_add (n m : Nat) : n.succ + m = (n + m).succ := by simp

theorem add_one (n : Nat) : n + 1 = n.succ := by simp

theorem one_add (n : Nat) : 1 + n = n.succ := by simp

theorem mul_succ (n m : Nat) : n * m.succ = n * m + n := by simp

theorem succ_mul (n m : Nat) : n.succ * m = n * m + m := by simp

theorem zero_lt_succ (n : Nat) : 0 < n.succ := by simp

theorem succ_le_succ (n m : Nat) : n.succ ≤ m.succ ↔ n ≤ m := by simp

theorem lt_succ_self (n : Nat) : n < n.succ := by simp

theorem le_refl (n : Nat) : n ≤ n := by simp

theorem le_trans {n m k : Nat} (h1 : n ≤ m) (h2 : m ≤ k) : n ≤ k := by simp [Nat.le_trans h1 h2]

theorem min_self (n : Nat) : min n n = n := by simp

theorem max_self (n : Nat) : max n n = n := by simp
"""

        # Test 5: Simple Equality
        simple_eq = """-- SimpleEq.lean
-- Test simp performance on equality reasoning

theorem eq_self (a : α) : a = a := by simp

theorem eq_comm {a b : α} (h : a = b) : b = a := by simp [h]

theorem eq_trans {a b c : α} (h1 : a = b) (h2 : b = c) : a = c := by simp [h1, h2]

theorem if_true (a b : α) : (if True then a else b) = a := by simp

theorem if_false (a b : α) : (if False then a else b) = b := by simp

theorem if_self (c : Prop) [Decidable c] (a : α) : (if c then a else a) = a := by simp

theorem ite_eq_left_iff (c : Prop) [Decidable c] (a b : α) : 
  (if c then a else b) = a ↔ c ∨ a = b := by
  split <;> simp [*]

theorem ite_eq_right_iff (c : Prop) [Decidable c] (a b : α) : 
  (if c then a else b) = b ↔ ¬c ∨ a = b := by
  split <;> simp [*]

theorem eq_rec_constant {α : Sort u} {a b : α} (h : a = b) (x : β) :
  @Eq.rec α a (fun _ => β) x b h = x := by simp

theorem cast_eq {α : Sort u} (h : α = α) (a : α) : cast h a = a := by simp

theorem heq_self (a : α) : HEq a a := by simp

theorem eq_mp_eq_cast {α β : Sort u} (h : α = β) : Eq.mp h = cast h := by simp

theorem eq_mpr_eq_cast {α β : Sort u} (h : α = β) : Eq.mpr h = cast h.symm := by simp

theorem cast_cast {α β γ : Sort u} (h1 : α = β) (h2 : β = γ) (a : α) :
  cast h2 (cast h1 a) = cast (h1.trans h2) a := by simp
"""

        # Write test files
        test_files = {
            "SimpleLists.lean": simple_lists,
            "BasicAlgebra.lean": basic_algebra,
            "LogicProofs.lean": logic_proofs,
            "BasicNat.lean": basic_nat,
            "SimpleEq.lean": simple_eq,
        }

        for filename, content in test_files.items():
            filepath = self.test_dir / filename
            filepath.write_text(content)
            print(f"Created {filepath}")

    def run_lean_with_profile(self, lean_file: Path) -> Dict:
        """Run Lean on a file with profiling enabled."""
        result = {
            "file": lean_file.name,
            "success": False,
            "compile_time": 0.0,
            "error": None,
            "simp_stats": {},
        }

        try:
            # Run Lean with profiling
            start_time = time.time()
            proc = subprocess.run(
                ["lake", "env", "lean", "--profile", str(lean_file)],
                capture_output=True,
                text=True,
                cwd=self.lean_dir,
            )
            end_time = time.time()

            result["compile_time"] = end_time - start_time
            result["success"] = proc.returncode == 0

            if proc.returncode != 0:
                result["error"] = proc.stderr
                print(f"Error compiling {lean_file.name}: {proc.stderr}")

            # Always try to extract simp-related information from output
            # (profiling data is available even when compilation "fails")
            result["simp_stats"] = self._parse_profile_output(proc.stdout, proc.stderr)

        except Exception as e:
            result["error"] = str(e)
            print(f"Exception running Lean on {lean_file.name}: {e}")

        return result

    def _parse_profile_output(self, stdout: str, stderr: str) -> Dict:
        """Parse profiler output for simp-related metrics."""
        stats = {
            "profile_lines": 0,
            "simp_mentions": 0,
            "tactic_mentions": 0,
            "simp_time_ms": 0.0,
            "tactic_execution_time_ms": 0.0,
            "elaboration_time_ms": 0.0,
            "total_import_time_ms": 0.0,
            "typeclass_inference_time_ms": 0.0,
        }

        # Parse profiling output to extract actual timing data
        full_output = stdout + "\n" + stderr
        for line in full_output.split("\n"):
            line = line.strip()
            if not line:
                continue

            stats["profile_lines"] += 1

            # Extract actual simp timing
            if line.startswith("simp "):
                time_match = re.search(r"simp (\d+(?:\.\d+)?)ms", line)
                if time_match:
                    stats["simp_time_ms"] = float(time_match.group(1))
                    stats["simp_mentions"] += 1

            # Extract tactic execution timing
            if line.startswith("tactic execution "):
                time_match = re.search(r"tactic execution (\d+(?:\.\d+)?)ms", line)
                if time_match:
                    stats["tactic_execution_time_ms"] = float(time_match.group(1))
                    stats["tactic_mentions"] += 1

            # Extract elaboration timing
            if line.startswith("elaboration "):
                time_match = re.search(r"elaboration (\d+(?:\.\d+)?)ms", line)
                if time_match:
                    stats["elaboration_time_ms"] = float(time_match.group(1))

            # Extract import timing
            if line.startswith("import took "):
                time_match = re.search(r"import took (\d+(?:\.\d+)?)(?:ms|s)", line)
                if time_match:
                    time_val = float(time_match.group(1))
                    if "s" in line:
                        time_val *= 1000  # Convert seconds to milliseconds
                    stats["total_import_time_ms"] = time_val

            # Extract typeclass inference timing
            if line.startswith("typeclass inference "):
                time_match = re.search(r"typeclass inference (\d+(?:\.\d+)?)ms", line)
                if time_match:
                    stats["typeclass_inference_time_ms"] = float(time_match.group(1))

        return stats

    def run_benchmarks(self) -> Dict:
        """Run all benchmarks and collect results."""
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "lean_version": self._get_lean_version(),
            "benchmarks": [],
        }

        # Test files should already exist in lean4/Benchmark/
        # self.create_test_files()

        # Run benchmarks on each file
        for lean_file in sorted(self.test_dir.glob("*.lean")):
            print(f"\nBenchmarking {lean_file.name}...")
            result = self.run_lean_with_profile(lean_file)
            results["benchmarks"].append(result)

            # Print summary
            if result["success"]:
                print(f"  ✓ Compiled in {result['compile_time']:.3f}s")
                print(f"  Profile lines: {result['simp_stats']['profile_lines']}")
                print(f"  Simp time: {result['simp_stats']['simp_time_ms']:.1f}ms")
                print(
                    f"  Tactic execution: {result['simp_stats']['tactic_execution_time_ms']:.1f}ms"
                )
                print(f"  Elaboration: {result['simp_stats']['elaboration_time_ms']:.1f}ms")
            else:
                print(f"  ✗ Failed to compile")
                # Still show timing data even if compilation "failed" (profiling data is still useful)
                stats = result["simp_stats"]
                if stats.get("simp_time_ms", 0) > 0:
                    print(f"  Simp time: {stats['simp_time_ms']:.1f}ms")
                if stats.get("tactic_execution_time_ms", 0) > 0:
                    print(f"  Tactic execution: {stats['tactic_execution_time_ms']:.1f}ms")
                if stats.get("elaboration_time_ms", 0) > 0:
                    print(f"  Elaboration: {stats['elaboration_time_ms']:.1f}ms")

        return results

    def _get_lean_version(self) -> str:
        """Get Lean version information."""
        try:
            proc = subprocess.run(
                ["lake", "env", "lean", "--version"],
                capture_output=True,
                text=True,
                cwd=self.lean_dir,
            )
            if proc.returncode == 0:
                return proc.stdout.strip()
        except:
            pass
        return "unknown"


def main():
    """Run the benchmark suite."""
    project_root = Path(__file__).parent.parent
    benchmark = LeanBenchmark(project_root)

    print("Running Lean 4 simp benchmarks...")
    print("=" * 60)

    results = benchmark.run_benchmarks()

    # Save results
    output_file = project_root / "benchmarks" / "baseline_measurements.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to {output_file}")

    # Print summary
    print("\nSummary:")
    print("-" * 40)
    total_time = sum(b["compile_time"] for b in results["benchmarks"])
    successful = sum(1 for b in results["benchmarks"] if b["success"])
    print(f"Total benchmarks: {len(results['benchmarks'])}")
    print(f"Successful: {successful}")
    print(f"Total compilation time: {total_time:.3f}s")
    print(f"Average time per file: {total_time/len(results['benchmarks']):.3f}s")


if __name__ == "__main__":
    main()
