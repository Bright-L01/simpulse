/-
Copyright (c) 2014 Parikshit Khanna. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Parikshit Khanna, Jeremy Avigad, Leonardo de Moura, Floris van Doorn, Mario Carneiro
-/
import Mathlib.Control.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Option.Basic
import Mathlib.Data.List.Defs
import Mathlib.Data.List.Monad
import Mathlib.Logic.OpClass
import Mathlib.Logic.Unique
import Mathlib.Order.Basic
import Mathlib.Tactic.Common

/-!
# Basic properties of lists
-/

assert_not_exists GroupWithZero
assert_not_exists Lattice
assert_not_exists Prod.swap_eq_iff_eq_swap
assert_not_exists Ring
assert_not_exists Set.range

open Function

open Nat hiding one_pos

namespace List

universe u v w

variable {ι : Type*} {α : Type u} {β : Type v} {γ : Type w} {l₁ l₂ : List α}

/-- There is only one list of an empty type -/
instance uniqueOfIsEmpty [IsEmpty α] : Unique (List α) :=
  { instInhabitedList with
    uniq := fun l =>
      match l with
      | [] => rfl
      | a :: _ => isEmptyElim a }

instance : Std.LawfulIdentity (α := List α) Append.append [] where
  left_id := nil_append
  right_id := append_nil

instance : Std.Associative (α := List α) Append.append where
  assoc := append_assoc

@[simp] theorem cons_injective {a : α} : Injective (cons a) := fun _ _ => tail_eq_of_cons_eq

theorem singleton_injective : Injective fun a : α => [a] := fun _ _ h => (cons_eq_cons.1 h).1

theorem set_of_mem_cons (l : List α) (a : α) : { x | x ∈ a :: l } = insert a { x | x ∈ l } :=
  Set.ext fun _ => mem_cons

/-! ### mem -/

theorem _root_.Decidable.List.eq_or_ne_mem_of_mem [DecidableEq α]
    {a b : α} {l : List α} (h : a ∈ b :: l) : a = b ∨ a ≠ b ∧ a ∈ l := by
  by_cases hab : a = b
  · exact Or.inl hab
  · exact ((List.mem_cons.1 h).elim Or.inl (fun h => Or.inr ⟨hab, h⟩))

lemma mem_pair {a b c : α} : a ∈ [b, c] ↔ a = b ∨ a = c := by
  rw [mem_cons, mem_singleton]


-- The simpNF linter says that the LHS can be simplified via `List.mem_map`.
-- However this is a higher priority lemma.
-- It seems the side condition `hf` is not applied by `simpNF`.
-- https://github.com/leanprover/std4/issues/207
@[simp 1100, nolint simpNF]
theorem mem_map_of_injective {f : α → β} (H : Injective f) {a : α} {l : List α} :
    f a ∈ map f l ↔ a ∈ l :=
  ⟨fun m => let ⟨_, m', e⟩ := exists_of_mem_map m; H e ▸ m', mem_map_of_mem⟩

@[simp]
theorem _root_.Function.Involutive.exists_mem_and_apply_eq_iff {f : α → α}
    (hf : Function.Involutive f) (x : α) (l : List α) : (∃ y : α, y ∈ l ∧ f y = x) ↔ f x ∈ l :=
  ⟨by rintro ⟨y, h, rfl⟩; rwa [hf y], fun h => ⟨f x, h, hf _⟩⟩

theorem mem_map_of_involutive {f : α → α} (hf : Involutive f) {a : α} {l : List α} :
    a ∈ map f l ↔ f a ∈ l := by rw [mem_map, hf.exists_mem_and_apply_eq_iff]

/-! ### length -/

alias ⟨_, length_pos_of_ne_nil⟩ := length_pos_iff

theorem length_pos_iff_ne_nil {l : List α} : 0 < length l ↔ l ≠ [] :=
  ⟨ne_nil_of_length_pos, length_pos_of_ne_nil⟩

theorem exists_of_length_succ {n} : ∀ l : List α, l.length = n + 1 → ∃ h t, l = h :: t
  | [], H => absurd H.symm <| succ_ne_zero n
  | h :: t, _ => ⟨h, t, rfl⟩

@[simp] lemma length_injective_iff : Injective (List.length : List α → ℕ) ↔ Subsingleton α := by
  constructor
  · intro h; refine ⟨fun x y => ?_⟩; (suffices [x] = [y] by simpa using this); apply h; rfl
  · intros hα l1 l2 hl
    induction l1 generalizing l2 <;> cases l2
    · rfl
    · cases hl
    · cases hl
    · next ih _ _ =>
      congr
      · subsingleton
      · apply ih; simpa using hl

@[simp default+1] -- Raise priority above `length_injective_iff`.
lemma length_injective [Subsingleton α] : Injective (length : List α → ℕ) :=
  length_injective_iff.mpr inferInstance

theorem length_eq_two {l : List α} : l.length = 2 ↔ ∃ a b, l = [a, b] :=
  ⟨fun _ => let [a, b] := l; ⟨a, b, rfl⟩, fun ⟨_, _, e⟩ => e ▸ rfl⟩

theorem length_eq_three {l : List α} : l.length = 3 ↔ ∃ a b c, l = [a, b, c] :=
  ⟨fun _ => let [a, b, c] := l; ⟨a, b, c, rfl⟩, fun ⟨_, _, _, e⟩ => e ▸ rfl⟩

/-! ### set-theoretic notation of lists -/

instance instSingletonList : Singleton α (List α) := ⟨fun x => [x]⟩

instance [DecidableEq α] : Insert α (List α) := ⟨List.insert⟩

instance [DecidableEq α] : LawfulSingleton α (List α) :=
  { insert_empty_eq := fun x =>
      show (if x ∈ ([] : List α) then [] else [x]) = [x] from if_neg not_mem_nil }

theorem singleton_eq (x : α) : ({x} : List α) = [x] :=
  rfl

theorem insert_neg [DecidableEq α] {x : α} {l : List α} (h : x ∉ l) :
    Insert.insert x l = x :: l :=
  insert_of_not_mem h

theorem insert_pos [DecidableEq α] {x : α} {l : List α} (h : x ∈ l) : Insert.insert x l = l :=
  insert_of_mem h

theorem doubleton_eq [DecidableEq α] {x y : α} (h : x ≠ y) : ({x, y} : List α) = [x, y] := by
  rw [insert_neg, singleton_eq]
  rwa [singleton_eq, mem_singleton]

/-! ### bounded quantifiers over lists -/

theorem forall_mem_of_forall_mem_cons {p : α → Prop} {a : α} {l : List α} (h : ∀ x ∈ a :: l, p x) :
    ∀ x ∈ l, p x := (forall_mem_cons.1 h).2

theorem exists_mem_cons_of {p : α → Prop} {a : α} (l : List α) (h : p a) : ∃ x ∈ a :: l, p x :=
  ⟨a, mem_cons_self, h⟩

theorem exists_mem_cons_of_exists {p : α → Prop} {a : α} {l : List α} : (∃ x ∈ l, p x) →
    ∃ x ∈ a :: l, p x :=
  fun ⟨x, xl, px⟩ => ⟨x, mem_cons_of_mem _ xl, px⟩

theorem or_exists_of_exists_mem_cons {p : α → Prop} {a : α} {l : List α} : (∃ x ∈ a :: l, p x) →
    p a ∨ ∃ x ∈ l, p x :=
  fun ⟨x, xal, px⟩ =>
    Or.elim (eq_or_mem_of_mem_cons xal) (fun h : x = a => by rw [← h]; left; exact px)
      fun h : x ∈ l => Or.inr ⟨x, h, px⟩

theorem exists_mem_cons_iff (p : α → Prop) (a : α) (l : List α) :
    (∃ x ∈ a :: l, p x) ↔ p a ∨ ∃ x ∈ l, p x :=
  Iff.intro or_exists_of_exists_mem_cons fun h =>
    Or.elim h (exists_mem_cons_of l) exists_mem_cons_of_exists

/-! ### list subset -/

theorem cons_subset_of_subset_of_mem {a : α} {l m : List α}
    (ainm : a ∈ m) (lsubm : l ⊆ m) : a::l ⊆ m :=
  cons_subset.2 ⟨ainm, lsubm⟩

theorem append_subset_of_subset_of_subset {l₁ l₂ l : List α} (l₁subl : l₁ ⊆ l) (l₂subl : l₂ ⊆ l) :
    l₁ ++ l₂ ⊆ l :=
  fun _ h ↦ (mem_append.1 h).elim (@l₁subl _) (@l₂subl _)

theorem map_subset_iff {l₁ l₂ : List α} (f : α → β) (h : Injective f) :
    map f l₁ ⊆ map f l₂ ↔ l₁ ⊆ l₂ := by
  refine ⟨?_, map_subset f⟩; intro h2 x hx
  rcases mem_map.1 (h2 (mem_map_of_mem hx)) with ⟨x', hx', hxx'⟩
  cases h hxx'; exact hx'

/-! ### append -/

theorem append_eq_has_append {L₁ L₂ : List α} : List.append L₁ L₂ = L₁ ++ L₂ :=
  rfl

theorem append_right_injective (s : List α) : Injective fun t ↦ s ++ t :=
  fun _ _ ↦ append_cancel_left

theorem append_left_injective (t : List α) : Injective fun s ↦ s ++ t :=
  fun _ _ ↦ append_cancel_right

/-! ### replicate -/

theorem eq_replicate_length {a : α} : ∀ {l : List α}, l = replicate l.length a ↔ ∀ b ∈ l, b = a
  | [] => by simp
  | (b :: l) => by simp [eq_replicate_length, replicate_succ]

