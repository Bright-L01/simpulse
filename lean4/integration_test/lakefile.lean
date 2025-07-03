import Lake
open Lake DSL

package «integration_test» where
  -- add package configuration options here

lean_lib «IntegrationTest» where
  -- add library configuration options here

@[default_target]
lean_exe «integration_test» where
  root := `Main