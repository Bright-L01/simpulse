import Lake
open Lake DSL

package simpulse where
  version := v!"1.1.0"
  keywords := #["optimization", "simp", "performance"]
  description := "High-performance optimization for Lean 4 simp tactics"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "main"

@[default_target]
lean_lib Simpulse where
  roots := #[`Simpulse]

-- Core library
lean_lib «Simpulse.Core» where
  roots := #[`Simpulse.Core]

-- Analysis library  
lean_lib «Simpulse.Analyzer» where
  roots := #[`Simpulse.Analyzer]

-- Optimization library
lean_lib «Simpulse.Optimizer» where
  roots := #[`Simpulse.Optimizer]

-- Lake integration
lean_lib «Simpulse.Integration» where
  roots := #[`Simpulse.Integration]

-- Apply library
lean_lib «Simpulse.Apply» where
  roots := #[`Simpulse.Apply]

-- Example executable
lean_exe demo where
  root := `Main
  supportInterpreter := true

-- Benchmark executable
lean_exe benchmark where
  root := `Benchmark.Main
  supportInterpreter := true

-- Test executable
lean_exe test where
  root := `Test.Main
  supportInterpreter := true