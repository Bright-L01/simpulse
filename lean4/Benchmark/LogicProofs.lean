-- LogicProofs.lean
-- Test simp performance on propositional logic

theorem and_comm (p q : Prop) : p ∧ q ↔ q ∧ p := by simp [and_comm]

theorem or_comm (p q : Prop) : p ∨ q ↔ q ∨ p := by simp [or_comm]

theorem and_assoc (p q r : Prop) : (p ∧ q) ∧ r ↔ p ∧ (q ∧ r) := by simp [and_assoc]

theorem or_assoc (p q r : Prop) : (p ∨ q) ∨ r ↔ p ∨ (q ∨ r) := by simp [or_assoc]

theorem not_not (p : Prop) [Decidable p] : ¬¬p ↔ p := by simp

theorem and_true (p : Prop) : p ∧ True ↔ p := by simp

theorem true_and (p : Prop) : True ∧ p ↔ p := by simp

theorem or_false (p : Prop) : p ∨ False ↔ p := by simp

theorem false_or (p : Prop) : False ∨ p ↔ p := by simp

theorem and_false (p : Prop) : p ∧ False ↔ False := by simp

theorem false_and (p : Prop) : False ∧ p ↔ False := by simp

theorem or_true (p : Prop) : p ∨ True ↔ True := by simp

theorem true_or (p : Prop) : True ∨ p ↔ True := by simp

theorem imp_self (p : Prop) : (p → p) ↔ True := by simp
