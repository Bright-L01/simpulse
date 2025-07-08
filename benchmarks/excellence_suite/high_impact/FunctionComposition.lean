-- FunctionComposition.lean
-- Function composition and identity operations with heavy simp usage
-- Expected: 2x+ speedup from frequent function simplification rules

-- Core function rules (very frequently used)
@[simp] theorem function_comp_id_left (f : α → β) : id ∘ f = f := by
  ext x; simp [Function.comp, id]

@[simp] theorem function_comp_id_right (f : α → β) : f ∘ id = f := by
  ext x; simp [Function.comp, id]

@[simp] theorem function_id_apply (x : α) : id x = x := by
  simp [id]

@[simp] theorem function_comp_apply (f : β → γ) (g : α → β) (x : α) : 
  (f ∘ g) x = f (g x) := by
  simp [Function.comp]

-- Composition associativity (frequently used)
@[simp] theorem function_comp_assoc (f : γ → δ) (g : β → γ) (h : α → β) :
  (f ∘ g) ∘ h = f ∘ (g ∘ h) := by
  ext x; simp [Function.comp]

-- Constant function rules (moderately used)
@[simp] theorem function_const_apply (b : β) (x : α) : 
  Function.const α b x = b := by
  simp [Function.const]

@[simp] theorem function_comp_const (f : β → γ) (b : β) :
  f ∘ Function.const α b = Function.const α (f b) := by
  ext x; simp [Function.comp, Function.const]

-- Heavy usage patterns (simulates real functional programming)
example (f : α → β) (g : β → γ) (h : γ → δ) :
  (h ∘ g) ∘ (f ∘ id) = h ∘ g ∘ f := by simp

example (f : α → β) :
  f ∘ id ∘ id = f := by simp

example (f : α → β) (g : β → γ) :
  (g ∘ f) ∘ id = g ∘ f := by simp

example (f : α → β) (g : β → γ) (h : γ → δ) (i : δ → ε) :
  ((i ∘ h) ∘ g) ∘ f = i ∘ (h ∘ (g ∘ f)) := by simp

-- Function application chains
example (f₁ : α → β) (f₂ : β → γ) (f₃ : γ → δ) (x : α) :
  (f₃ ∘ f₂ ∘ f₁) x = f₃ (f₂ (f₁ x)) := by simp

example (f : α → β) (x : α) :
  (f ∘ id) x = f x := by simp

example (f : α → β) (g : β → γ) (x : α) :
  ((g ∘ f) ∘ id) x = g (f x) := by simp

-- Constant function patterns
example (c : γ) (f : α → β) :
  Function.const β c ∘ f = Function.const α c := by simp

example (c : β) (x : α) :
  (Function.const α c ∘ id) x = c := by simp

-- Map operations on functors (when available)
@[simp] theorem list_map_id (l : List α) : l.map id = l := by
  induction l with
  | nil => simp
  | cons head tail ih => simp [List.map, ih]

@[simp] theorem list_map_comp (f : β → γ) (g : α → β) (l : List α) :
  l.map (f ∘ g) = (l.map g).map f := by
  induction l with
  | nil => simp
  | cons head tail ih => simp [List.map, ih]

-- Option map patterns
@[simp] theorem option_map_id (o : Option α) : o.map id = o := by
  cases o <;> simp [Option.map]

@[simp] theorem option_map_comp (f : β → γ) (g : α → β) (o : Option α) :
  o.map (f ∘ g) = (o.map g).map f := by
  cases o <;> simp [Option.map]

-- Complex composition chains
example (f₁ : α → β) (f₂ : β → γ) (f₃ : γ → δ) (f₄ : δ → ε) (f₅ : ε → ζ) :
  f₅ ∘ f₄ ∘ f₃ ∘ f₂ ∘ f₁ ∘ id = f₅ ∘ (f₄ ∘ (f₃ ∘ (f₂ ∘ f₁))) := by simp

example (f : α → β) (g : β → γ) :
  (g ∘ f) ∘ id ∘ id = g ∘ f := by simp

-- Function pipeline patterns (common in functional style)
example (data : List α) (f : α → β) (g : β → γ) (h : γ → δ) :
  data.map f |>.map g |>.map h = data.map (h ∘ g ∘ f) := by simp

example (opt : Option α) (f : α → β) (g : β → γ) :
  opt.map f |>.map g = opt.map (g ∘ f) := by simp

-- Identity elimination in complex expressions
example (f : α → β) (g : β → γ) (h : γ → δ) :
  h ∘ (g ∘ f ∘ id) ∘ id = h ∘ g ∘ f := by simp

example (funcs : List (α → α)) :
  funcs.map (· ∘ id) = funcs := by simp

-- Curry/uncurry patterns (when available)
example (f : α × β → γ) (a : α) (b : β) :
  Function.curry f a b = f (a, b) := by simp [Function.curry]

example (f : α → β → γ) (p : α × β) :
  Function.uncurry f p = f p.1 p.2 := by simp [Function.uncurry]

-- Performance stress patterns
example (l : List α) (f₁ : α → β) (f₂ : β → γ) (f₃ : γ → δ) (f₄ : δ → ε) :
  l.map f₁ |>.map f₂ |>.map f₃ |>.map f₄ = l.map (f₄ ∘ f₃ ∘ f₂ ∘ f₁) := by simp

example (o : Option α) (f : α → α) :
  o.map f |>.map id |>.map f = o.map (f ∘ f) := by simp

-- Very frequent micro-patterns in functional code
example (f : α → β) : f ∘ id = f := by simp
example (f : α → β) : id ∘ f = f := by simp
example (x : α) : id x = x := by simp
example (f : α → β) (g : β → γ) (x : α) : (g ∘ f) x = g (f x) := by simp

-- Edge cases
example : (id : α → α) ∘ id = id := by simp
example (c : β) : Function.const α c ∘ id = Function.const α c := by simp
example (f : α → β) : f ∘ id ∘ id ∘ id = f := by simp