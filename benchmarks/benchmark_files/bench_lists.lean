
-- Benchmark: List operations
example (l : List α) : l ++ [] = l := by simp
example (l : List α) : [] ++ l = l := by simp
example (l : List α) (a : α) : (a :: l).length = l.length + 1 := by simp
