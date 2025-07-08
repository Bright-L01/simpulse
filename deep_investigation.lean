-- Deep investigation of what lemmas exist in core Lean 4

-- These exist in CORE Lean (not Mathlib!)
#check @Nat.add_zero    -- n + 0 = n
#check @Nat.zero_add    -- 0 + n = n  
#check @Nat.mul_one     -- n * 1 = n
#check @Nat.one_mul     -- 1 * n = n
#check @Nat.zero_mul    -- 0 * n = 0
#check @Nat.mul_zero    -- n * 0 = 0

-- Check if List lemmas exist
#check @List.map_cons   -- This might not exist
#check @List.append_nil -- This might not exist

-- Boolean/logic lemmas
#check @eq_self_iff_true
#check @true_and
#check @and_true