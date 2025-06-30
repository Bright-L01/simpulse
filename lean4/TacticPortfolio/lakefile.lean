import Lake
open Lake DSL

package «tactic-portfolio» where
  -- add package configuration options here

lean_lib «TacticPortfolio» where
  -- add library configuration options here

@[default_target]
lean_exe «tactic-portfolio» where
  root := `Main
  supportInterpreter := true