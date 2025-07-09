-- Main.lean
-- Main entry point

import Simpulse

#check "Main loaded"

def main : IO Unit := do
  IO.println "Simpulse demo"