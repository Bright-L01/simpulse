/-
  Integration mechanism for Simpulse JIT
  
  Provides transparent integration with existing Lean code
  through tactic modification and environment variables.
-/

import Lean
import SimpulseJIT.Profiler

open Lean Elab Tactic
open SimpulseJIT

namespace SimpulseJIT.Integration

/-- Check if JIT profiling is enabled via environment variable -/
def isJITEnabled : IO Bool := do
  let env ← IO.getEnv "SIMPULSE_JIT"
  return env == some "1" || env == some "true"

/-- Load saved priorities from disk -/
def loadSavedPriorities (path : String) : IO (Option (HashMap Name Nat)) := do
  try
    let content ← IO.FS.readFile path
    -- Parse simple JSON format
    let lines := content.splitOn "\n" |>.filter (·.contains ":")
    let priorities := lines.foldl (init := HashMap.empty) fun acc line =>
      match line.splitOn ":" with
      | [name, prio] =>
        let cleanName := name.trim.removePrefix "\"" |>.removeSuffix "\""
        let cleanPrio := prio.trim.removeSuffix ","
        match cleanPrio.toNat? with
        | some p => acc.insert cleanName.toName p
        | none => acc
      | _ => acc
    return some priorities
  catch _ =>
    return none

/-- Modified simp tactic that uses JIT profiling when enabled -/
@[tactic simp]
def simpJIT : Tactic := fun stx => do
  let enabled ← isJITEnabled
  
  if enabled then
    -- Initialize profiler with config from environment
    let logFile ← IO.getEnv "SIMPULSE_JIT_LOG"
    let savePath ← IO.getEnv "SIMPULSE_JIT_SAVE" <|> pure "simp_priorities.json"
    let interval := (← IO.getEnv "SIMPULSE_JIT_INTERVAL").bind (·.toNat?) |>.getD 100
    
    let config : ProfilerConfig := {
      enabled := true
      logFile := logFile
      adaptationInterval := interval
      savePath := some savePath
    }
    
    enableProfiling config
    
    -- Load any saved priorities
    match ← loadSavedPriorities savePath with
    | some priorities =>
      IO.println s!"Loaded {priorities.size} saved priorities"
      -- TODO: Apply loaded priorities to simp theorems
    | none => pure ()
    
    -- Run simp with profiling
    simpWithProfiling
  else
    -- Run standard simp
    evalTactic (← `(tactic| simp))

/-- Tactic to display current JIT statistics -/
syntax (name := simpStats) "simp_stats" : tactic

@[tactic simpStats]
def evalSimpStats : Tactic := fun _ => do
  let summary ← getStatsSummary
  logInfo summary

/-- Tactic to manually trigger priority optimization -/
syntax (name := simpOptimize) "simp_optimize" : tactic

@[tactic simpOptimize]  
def evalSimpOptimize : Tactic := fun _ => do
  let priorities ← optimizePriorities
  logInfo s!"Optimized {priorities.size} rule priorities"
  
  -- Save to disk if configured
  match (← profilerConfig.get).savePath with
  | some path =>
    let json := priorities.toList.map fun (name, prio) => 
      s!"\"{name}\": {prio}"
    let content := "{\n  " ++ String.intercalate ",\n  " json ++ "\n}"
    IO.FS.writeFile path content
    logInfo s!"Saved priorities to {path}"
  | none => pure ()

/-- Command to enable JIT profiling for the current file -/
elab "enable_jit_profiling" : command => do
  enableProfiling
  logInfo "JIT profiling enabled for this file"

/-- Command to show JIT statistics -/
elab "#simp_stats" : command => do
  let summary ← getStatsSummary
  logInfo summary

/-- Attribute to mark theorems for priority tracking -/
syntax (name := track_priority) "track_priority" : attr

initialize registerBuiltinAttribute {
  name := `track_priority
  descr := "Mark theorem for Simpulse JIT priority tracking"
  add := fun _ _ _ => pure ()
}

end SimpulseJIT.Integration