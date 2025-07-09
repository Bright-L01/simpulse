/-
  Demo of Tactic Portfolio
  
  Shows how the portfolio tactic selects appropriate tactics
  based on goal structure.
-/

import TacticPortfolio.Portfolio

open TacticPortfolio

-- Configure portfolio for demo
def demoConfig : PortfolioConfig := {
  logAttempts := true
  maxAttempts := 3
  useML := true
}

-- Example 1: Simple arithmetic (should use simp)
example (x : Nat) : x + 0 = x := by
  portfolio

-- Example 2: Polynomial equation (should use ring)
example (a b : Int) : (a + b)^2 = a^2 + 2*a*b + b^2 := by
  portfolio

-- Example 3: Linear inequality (should use linarith)
example (x y : Real) (h : x < y) : x < y + 1 := by
  portfolio

-- Example 4: Numerical computation (should use norm_num)
example : 42 * 17 = 714 := by
  portfolio

-- Example 5: Mixed - portfolio should try multiple tactics
example (x y : Nat) (h : x = y + 5) : x + 0 = y + 5 := by
  portfolio

-- Test the auto tactic
example (a b c : Real) : a * (b + c) = a * b + a * c := by
  auto

example (x : Int) : x < x + 1 := by
  auto

-- Show accumulated statistics
#portfolio_stats

-- More complex example requiring tactic combination
example (x y z : Real) (h1 : x < y) (h2 : y < z) : 
  x + 0 < z + 1 := by
  portfolio  -- Will try simp first, then linarith

def main : IO Unit := do
  IO.println "Tactic Portfolio Demo"
  IO.println "===================="
  
  -- Export statistics for analysis
  exportStats "portfolio_stats.json"
  
  IO.println "\nThe portfolio tactic automatically selects:"
  IO.println "- simp for simplification"
  IO.println "- ring for polynomial equations"
  IO.println "- linarith for linear inequalities"
  IO.println "- norm_num for numerical computation"
  IO.println "- abel for abelian group equations"
  
  IO.println "\nBased on ML predictions from goal structure!"