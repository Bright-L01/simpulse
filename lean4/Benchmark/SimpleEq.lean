-- SimpleEq.lean
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
