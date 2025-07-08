-- SetOperations.lean
-- Set theory operations with heavy simp usage
-- Expected: 2x+ speedup from frequent set simplification rules

-- Core set rules (very frequently used)
@[simp] theorem set_union_empty (s : Set α) : s ∪ ∅ = s := by
  ext x; simp [Set.mem_union, Set.mem_empty]

@[simp] theorem empty_union_set (s : Set α) : ∅ ∪ s = s := by
  ext x; simp [Set.mem_union, Set.mem_empty]

@[simp] theorem set_inter_empty (s : Set α) : s ∩ ∅ = ∅ := by
  ext x; simp [Set.mem_inter, Set.mem_empty]

@[simp] theorem empty_inter_set (s : Set α) : ∅ ∩ s = ∅ := by
  ext x; simp [Set.mem_inter, Set.mem_empty]

@[simp] theorem set_union_univ (s : Set α) : s ∪ Set.univ = Set.univ := by
  ext x; simp [Set.mem_union, Set.mem_univ]

@[simp] theorem univ_union_set (s : Set α) : Set.univ ∪ s = Set.univ := by
  ext x; simp [Set.mem_union, Set.mem_univ]

@[simp] theorem set_inter_univ (s : Set α) : s ∩ Set.univ = s := by
  ext x; simp [Set.mem_inter, Set.mem_univ]

@[simp] theorem univ_inter_set (s : Set α) : Set.univ ∩ s = s := by
  ext x; simp [Set.mem_inter, Set.mem_univ]

-- Idempotency and self-operations (frequently used)
@[simp] theorem set_union_self (s : Set α) : s ∪ s = s := by
  ext x; simp [Set.mem_union]

@[simp] theorem set_inter_self (s : Set α) : s ∩ s = s := by
  ext x; simp [Set.mem_inter]

-- Complement operations (moderately used)
@[simp] theorem set_union_compl (s : Set α) : s ∪ sᶜ = Set.univ := by
  ext x; simp [Set.mem_union, Set.mem_compl, Classical.em]

@[simp] theorem set_inter_compl (s : Set α) : s ∩ sᶜ = ∅ := by
  ext x; simp [Set.mem_inter, Set.mem_compl, Set.mem_empty]

@[simp] theorem compl_empty : (∅ : Set α)ᶜ = Set.univ := by
  ext x; simp [Set.mem_compl, Set.mem_empty, Set.mem_univ]

@[simp] theorem compl_univ : (Set.univ : Set α)ᶜ = ∅ := by
  ext x; simp [Set.mem_compl, Set.mem_univ, Set.mem_empty]

-- Heavy usage patterns (simulates real set theory proofs)
example (s t u : Set α) : (s ∪ t) ∪ u ∪ ∅ = s ∪ t ∪ u := by simp [Set.union_assoc]

example (s t : Set α) : s ∪ ∅ ∪ t ∪ ∅ = s ∪ t := by simp

example (s t u : Set α) : (s ∩ t) ∩ u ∩ Set.univ = s ∩ t ∩ u := by simp [Set.inter_assoc]

example (s : Set α) : s ∪ ∅ ∪ ∅ = s := by simp

example (s t : Set α) : (s ∩ Set.univ) ∪ (t ∩ Set.univ) = s ∪ t := by simp

-- Distributivity patterns (common in algebra of sets)
example (s t u : Set α) : s ∩ (t ∪ u ∪ ∅) = (s ∩ t) ∪ (s ∩ u) := by simp [Set.inter_distrib_left]

example (s t u : Set α) : (s ∪ t ∪ ∅) ∩ u = (s ∩ u) ∪ (t ∩ u) := by simp [Set.inter_distrib_right]

-- De Morgan's laws with simplification
example (s t : Set α) : (s ∪ t ∪ ∅)ᶜ = sᶜ ∩ tᶜ := by simp [Set.compl_union]

example (s t : Set α) : (s ∩ t ∩ Set.univ)ᶜ = sᶜ ∪ tᶜ := by simp [Set.compl_inter]

-- Subset and membership simplifications
example (s : Set α) : ∅ ⊆ s := by simp [Set.empty_subset]

example (s : Set α) : s ⊆ Set.univ := by simp [Set.subset_univ]

example (x : α) (s : Set α) : x ∈ s ∪ ∅ ↔ x ∈ s := by simp

example (x : α) (s : Set α) : x ∈ s ∩ Set.univ ↔ x ∈ s := by simp

-- Set difference operations
@[simp] theorem set_diff_empty (s : Set α) : s \ ∅ = s := by
  ext x; simp [Set.mem_diff, Set.mem_empty]

@[simp] theorem empty_diff_set (s : Set α) : ∅ \ s = ∅ := by
  ext x; simp [Set.mem_diff, Set.mem_empty]

@[simp] theorem set_diff_self (s : Set α) : s \ s = ∅ := by
  ext x; simp [Set.mem_diff, Set.mem_empty]

@[simp] theorem set_diff_univ (s : Set α) : s \ Set.univ = ∅ := by
  ext x; simp [Set.mem_diff, Set.mem_univ, Set.mem_empty]

-- Complex set expressions (stress tests)
example (s t u v : Set α) : 
  ((s ∪ ∅) ∩ (t ∪ ∅)) ∪ ((u ∩ Set.univ) ∩ (v ∩ Set.univ)) = 
  (s ∩ t) ∪ (u ∩ v) := by simp

example (s t : Set α) :
  (s ∪ ∅) \ (∅ ∪ t) = s \ t := by simp

example (s t u : Set α) :
  (s ∩ Set.univ) ∪ (t ∩ Set.univ) ∪ (u ∪ ∅) = s ∪ t ∪ u := by simp

-- Image and preimage operations (frequent in function theory)
@[simp] theorem image_empty (f : α → β) : f '' ∅ = ∅ := by
  ext y; simp [Set.mem_image, Set.mem_empty]

@[simp] theorem preimage_empty (f : α → β) : f ⁻¹' ∅ = ∅ := by
  ext x; simp [Set.mem_preimage, Set.mem_empty]

@[simp] theorem preimage_univ (f : α → β) : f ⁻¹' Set.univ = Set.univ := by
  ext x; simp [Set.mem_preimage, Set.mem_univ]

-- Set operations with images
example (f : α → β) (s t : Set α) : 
  f '' (s ∪ t ∪ ∅) = f '' s ∪ f '' t := by simp [Set.image_union]

example (f : α → β) (s t : Set β) :
  f ⁻¹' (s ∩ t ∩ Set.univ) = f ⁻¹' s ∩ f ⁻¹' t := by simp [Set.preimage_inter]

-- Power set operations
example (s : Set α) : ∅ ∈ 𝒫 s := by simp [Set.mem_powerset, Set.empty_subset]

example (s : Set α) : s ∈ 𝒫 s := by simp [Set.mem_powerset]

-- Very frequent micro-patterns in set theory
example (s : Set α) : s ∪ ∅ = s := by simp
example (s : Set α) : ∅ ∪ s = s := by simp
example (s : Set α) : s ∩ Set.univ = s := by simp
example (s : Set α) : Set.univ ∩ s = s := by simp
example (s : Set α) : s ∩ ∅ = ∅ := by simp
example (s : Set α) : ∅ ∩ s = ∅ := by simp

-- Edge cases
example : (∅ : Set α) ∪ ∅ = ∅ := by simp
example : (Set.univ : Set α) ∩ Set.univ = Set.univ := by simp
example : (∅ : Set α) \ ∅ = ∅ := by simp