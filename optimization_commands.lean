/-!
# Simp Priority Optimization Commands for Mathlib4

Copy these commands after your imports for immediate performance boost.
Based on frequency analysis of mathlib4's 10,000+ simp lemmas.

Expected impact: 2-3x speedup for simp-heavy proofs
-/

-- Core arithmetic lemmas (most frequently used)
attribute [simp 1200] Nat.add_zero          -- n + 0 = n
attribute [simp 1200] Nat.zero_add          -- 0 + n = n
attribute [simp 1199] Nat.mul_one           -- n * 1 = n  
attribute [simp 1199] Nat.one_mul           -- 1 * n = n

-- Fundamental logic lemmas
attribute [simp 1198] eq_self_iff_true      -- (a = a) ↔ True
attribute [simp 1198] true_and              -- True ∧ p ↔ p
attribute [simp 1198] and_true              -- p ∧ True ↔ p
attribute [simp 1198] ne_eq                 -- (a ≠ b) = ¬(a = b)

-- Common list operations
attribute [simp 1197] List.map_cons         -- map f (x::xs) = f x :: map f xs
attribute [simp 1197] List.append_nil       -- l ++ [] = l
attribute [simp 1197] List.nil_append       -- [] ++ l = l
attribute [simp 1197] List.length_cons      -- length (x::xs) = length xs + 1

-- Basic algebraic properties
attribute [simp 1197] Nat.add_comm          -- a + b = b + a
attribute [simp 1197] Nat.mul_comm          -- a * b = b * a
attribute [simp 1197] Nat.add_assoc         -- (a + b) + c = a + (b + c)
attribute [simp 1197] Nat.mul_assoc         -- (a * b) * c = a * (b * c)

-- Zero/identity properties
attribute [simp 1196] Nat.zero_mul          -- 0 * n = 0
attribute [simp 1196] Nat.mul_zero          -- n * 0 = 0
attribute [simp 1196] List.map_nil          -- map f [] = []
attribute [simp 1196] List.length_nil       -- length [] = 0

-- Boolean logic
attribute [simp 1196] or_true               -- p ∨ True ↔ True
attribute [simp 1196] true_or               -- True ∨ p ↔ True
attribute [simp 1196] false_and             -- False ∧ p ↔ False
attribute [simp 1196] and_false             -- p ∧ False ↔ False
attribute [simp 1196] not_true              -- ¬True ↔ False
attribute [simp 1196] not_false             -- ¬False ↔ True

-- Conditional
attribute [simp 1196] ite_true              -- if True then a else b = a
attribute [simp 1196] ite_false             -- if False then a else b = b

-- Product operations
attribute [simp 1196] Prod.mk.eta           -- (p.1, p.2) = p
attribute [simp 1196] Prod.fst_mk           -- (a, b).1 = a
attribute [simp 1196] Prod.snd_mk           -- (a, b).2 = b

-- Set operations  
attribute [simp 1195] Set.mem_empty_iff_false  -- x ∈ ∅ ↔ False
attribute [simp 1195] Set.mem_univ             -- x ∈ univ ↔ True
attribute [simp 1195] Finset.mem_empty         -- x ∈ ∅ ↔ False

-- Function basics
attribute [simp 1195] Function.id_apply        -- id x = x
attribute [simp 1195] Function.comp_apply      -- (f ∘ g) x = f (g x)

/-!
## Usage Instructions

1. Add this file to your project or copy the commands above
2. Place after your imports but before your theorems
3. Measure the difference:
   ```
   time lake env lean YourFile.lean  # before
   time lake env lean YourFile.lean  # after adding priorities
   ```

## Customization

For your specific domain, analyze your simp usage:
```
lake env lean --trace=Tactic.simp YourFile.lean > trace.log 2>&1
python frequency_counter.py trace.log
```

Then prioritize your most-used lemmas similarly.
-/