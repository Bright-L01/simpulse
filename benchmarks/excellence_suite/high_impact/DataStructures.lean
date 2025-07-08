-- DataStructures.lean
-- Basic data structure operations with heavy simp usage
-- Expected: 2x+ speedup from frequent data structure simplification rules

-- Array operations (very frequently used)
@[simp] theorem array_size_empty : (#[] : Array α).size = 0 := by simp [Array.size]

@[simp] theorem array_get_push (arr : Array α) (x : α) (i : Fin arr.size) :
  (arr.push x)[i]'(Nat.lt_succ_of_lt i.isLt) = arr[i] := by simp [Array.get_push]

@[simp] theorem array_size_push (arr : Array α) (x : α) : 
  (arr.push x).size = arr.size + 1 := by simp [Array.size_push]

-- Vector operations (frequently used)
@[simp] theorem vector_length (v : Vector α n) : v.length = n := by simp [Vector.length]

@[simp] theorem vector_cons_length (x : α) (v : Vector α n) : 
  (Vector.cons x v).length = n + 1 := by simp [Vector.cons]

@[simp] theorem vector_nil_length : (@Vector.nil α).length = 0 := by simp [Vector.nil]

-- Finset operations (frequently used)
@[simp] theorem finset_card_empty : (∅ : Finset α).card = 0 := by simp [Finset.card_empty]

@[simp] theorem finset_mem_empty (x : α) : x ∉ (∅ : Finset α) := by simp [Finset.mem_empty]

@[simp] theorem finset_union_empty (s : Finset α) : s ∪ ∅ = s := by simp [Finset.union_empty]

@[simp] theorem finset_empty_union (s : Finset α) : ∅ ∪ s = s := by simp [Finset.empty_union]

@[simp] theorem finset_inter_empty (s : Finset α) : s ∩ ∅ = ∅ := by simp [Finset.inter_empty]

@[simp] theorem finset_empty_inter (s : Finset α) : ∅ ∩ s = ∅ := by simp [Finset.empty_inter]

-- Multiset operations (moderately used)
@[simp] theorem multiset_card_zero : (0 : Multiset α).card = 0 := by simp [Multiset.card_zero]

@[simp] theorem multiset_mem_zero (x : α) : x ∉ (0 : Multiset α) := by simp [Multiset.mem_zero]

@[simp] theorem multiset_add_zero (s : Multiset α) : s + 0 = s := by simp [Multiset.add_zero]

@[simp] theorem multiset_zero_add (s : Multiset α) : 0 + s = s := by simp [Multiset.zero_add]

-- Heavy usage patterns (simulates real data structure manipulation)
example (arr : Array Nat) (x y : Nat) :
  (arr.push x).push y |>.size = arr.size + 2 := by simp [Nat.add_assoc]

example (s t : Finset Nat) :
  (s ∪ ∅) ∩ (t ∪ ∅) = s ∩ t := by simp

example (v : Vector Nat n) (x y z : Nat) :
  (Vector.cons x (Vector.cons y (Vector.cons z v))).length = n + 3 := by simp [Nat.add_assoc]

-- Array manipulation chains
example (arr : Array String) (a b c : String) :
  arr.push a |>.push b |>.push c |>.size = arr.size + 3 := by simp [Nat.add_assoc]

example (arr : Array α) :
  (#[].push x).size = 1 := by simp

-- Finset union/intersection patterns
example (s₁ s₂ s₃ : Finset Nat) :
  (s₁ ∪ ∅) ∪ (s₂ ∪ ∅) ∪ (s₃ ∪ ∅) = s₁ ∪ s₂ ∪ s₃ := by simp [Finset.union_assoc]

example (s t : Finset α) :
  (s ∩ ∅) ∪ (t ∩ ∅) = ∅ := by simp

example (s : Finset Nat) :
  s ∪ ∅ ∪ ∅ = s := by simp

-- Vector construction patterns
example (x y : α) :
  (Vector.cons x (Vector.cons y Vector.nil)).length = 2 := by simp

example (v : Vector α n) (x : α) :
  (Vector.cons x v).length = v.length + 1 := by simp

-- Multiset addition chains
example (s₁ s₂ s₃ : Multiset Nat) :
  (s₁ + 0) + (s₂ + 0) + (s₃ + 0) = s₁ + s₂ + s₃ := by simp [Multiset.add_assoc]

example (s : Multiset α) :
  s + 0 + 0 = s := by simp

-- Hash map/dictionary patterns (when available)
@[simp] theorem hashmap_empty_size : (HashMap.empty : HashMap α β).size = 0 := by 
  simp [HashMap.size, HashMap.empty]

@[simp] theorem hashmap_mem_empty (k : α) : k ∉ (HashMap.empty : HashMap α β) := by
  simp [HashMap.mem, HashMap.empty]

-- Complex data structure expressions
example (arr₁ arr₂ : Array Nat) (x y : Nat) :
  (arr₁.push x ++ arr₂.push y).size = arr₁.size + arr₂.size + 2 := by simp [Array.size_append, Nat.add_assoc, Nat.add_comm]

example (s₁ s₂ s₃ s₄ : Finset String) :
  ((s₁ ∪ ∅) ∩ (s₂ ∪ ∅)) ∪ ((s₃ ∪ ∅) ∩ (s₄ ∪ ∅)) = 
  (s₁ ∩ s₂) ∪ (s₃ ∩ s₄) := by simp

-- Performance stress patterns
example (arrs : List (Array Nat)) :
  arrs.map (·.push 0) |>.map (·.size) = arrs.map (fun arr => arr.size + 1) := by simp

example (sets : List (Finset Nat)) :
  sets.map (· ∪ ∅) = sets := by simp

-- Tree-like structure operations (when available)
@[simp] theorem binary_tree_size_leaf : (@BinaryTree.leaf α).size = 1 := by
  simp [BinaryTree.size, BinaryTree.leaf]

@[simp] theorem binary_tree_size_node (left : BinaryTree α) (val : α) (right : BinaryTree α) :
  (BinaryTree.node left val right).size = left.size + right.size + 1 := by
  simp [BinaryTree.size, BinaryTree.node]

-- Stack operations (when available)
@[simp] theorem stack_empty_size : (@Stack.empty α).size = 0 := by simp [Stack.size, Stack.empty]

@[simp] theorem stack_push_size (s : Stack α) (x : α) : 
  (s.push x).size = s.size + 1 := by simp [Stack.size, Stack.push]

-- Queue operations (when available)  
@[simp] theorem queue_empty_size : (@Queue.empty α).size = 0 := by simp [Queue.size, Queue.empty]

@[simp] theorem queue_enqueue_size (q : Queue α) (x : α) :
  (q.enqueue x).size = q.size + 1 := by simp [Queue.size, Queue.enqueue]

-- Very frequent micro-patterns in data structure code
example (arr : Array α) : (#[].push x).size = 1 := by simp
example (s : Finset α) : s ∪ ∅ = s := by simp
example (s : Finset α) : ∅ ∪ s = s := by simp
example (s : Finset α) : s ∩ ∅ = ∅ := by simp
example (m : Multiset α) : m + 0 = m := by simp
example (v : Vector α n) (x : α) : (Vector.cons x v).length = n + 1 := by simp

-- Edge cases
example : (#[] : Array Nat).size = 0 := by simp
example : (∅ : Finset Nat).card = 0 := by simp
example : (@Vector.nil Nat).length = 0 := by simp
example : (0 : Multiset Nat).card = 0 := by simp