-- BasicCategory.lean
-- Simple category theory with heavy simp usage
-- Expected: 2x+ speedup from frequent categorical simplification rules

-- Category axioms (very frequently used)
@[simp] theorem category_id_comp {C : Type*} [Category C] (f : X ⟶ Y) : 
  𝟙 Y ≫ f = f := by simp [Category.id_comp]

@[simp] theorem category_comp_id {C : Type*} [Category C] (f : X ⟶ Y) : 
  f ≫ 𝟙 X = f := by simp [Category.comp_id]

@[simp] theorem category_assoc {C : Type*} [Category C] (f : W ⟶ X) (g : X ⟶ Y) (h : Y ⟶ Z) :
  (f ≫ g) ≫ h = f ≫ (g ≫ h) := by simp [Category.assoc]

-- Functor laws (frequently used)
@[simp] theorem functor_map_id {C D : Type*} [Category C] [Category D] (F : C ⥤ D) (X : C) :
  F.map (𝟙 X) = 𝟙 (F.obj X) := by simp [Functor.map_id]

@[simp] theorem functor_map_comp {C D : Type*} [Category C] [Category D] (F : C ⥤ D) 
    (f : X ⟶ Y) (g : Y ⟶ Z) :
  F.map (f ≫ g) = F.map f ≫ F.map g := by simp [Functor.map_comp]

-- Natural transformation axioms (moderately used)
@[simp] theorem nat_trans_naturality {C D : Type*} [Category C] [Category D] 
    (F G : C ⥤ D) (α : F ⟶ G) (f : X ⟶ Y) :
  F.map f ≫ α.app Y = α.app X ≫ G.map f := by simp [NatTrans.naturality]

-- Heavy usage patterns (simulates real category theory proofs)
variable {C D E : Type*} [Category C] [Category D] [Category E]

example (f : X ⟶ Y) (g : Y ⟶ Z) : 
  f ≫ g ≫ 𝟙 Z = f ≫ g := by simp

example (f : X ⟶ Y) : 
  𝟙 Y ≫ f ≫ 𝟙 X = f := by simp

example (f : W ⟶ X) (g : X ⟶ Y) (h : Y ⟶ Z) (i : Z ⟶ U) :
  ((f ≫ g) ≫ h) ≫ i = f ≫ (g ≫ (h ≫ i)) := by simp

-- Functor composition patterns
example (F : C ⥤ D) (G : D ⥤ E) (f : X ⟶ Y) :
  (F ⋙ G).map f = G.map (F.map f) := by simp [Functor.comp_map]

example (F : C ⥤ D) (X : C) :
  (F ⋙ 𝟭 D).map (𝟙 X) = 𝟙 (F.obj X) := by simp

example (F : C ⥤ D) (f : X ⟶ Y) (g : Y ⟶ Z) :
  F.map (𝟙 X ≫ f ≫ g) = F.map f ≫ F.map g := by simp

-- Identity functor patterns
@[simp] theorem id_functor_obj {C : Type*} [Category C] (X : C) :
  (𝟭 C).obj X = X := by simp [Functor.id_obj]

@[simp] theorem id_functor_map {C : Type*} [Category C] (f : X ⟶ Y) :
  (𝟭 C).map f = f := by simp [Functor.id_map]

example (f : X ⟶ Y) (g : Y ⟶ Z) :
  (𝟭 C).map (f ≫ g ≫ 𝟙 Z) = f ≫ g := by simp

-- Composition with identity morphisms
example (f : X ⟶ Y) (g : Y ⟶ Z) (h : Z ⟶ W) :
  f ≫ 𝟙 Y ≫ g ≫ 𝟙 Z ≫ h = f ≫ g ≫ h := by simp

example (F : C ⥤ D) (f : X ⟶ Y) :
  F.map (𝟙 X ≫ f ≫ 𝟙 Y) = F.map f := by simp

-- Natural transformation identity patterns
@[simp] theorem nat_trans_id_app {C D : Type*} [Category C] [Category D] (F : C ⥤ D) (X : C) :
  (𝟙 F : F ⟶ F).app X = 𝟙 (F.obj X) := by simp [NatTrans.id_app]

example (F G : C ⥤ D) (α : F ⟶ G) (X : C) :
  α.app X ≫ 𝟙 (G.obj X) = α.app X := by simp

example (F : C ⥤ D) (f : X ⟶ Y) :
  F.map f ≫ (𝟙 F : F ⟶ F).app Y = F.map f := by simp

-- Isomorphism patterns (frequent in equivalences)
@[simp] theorem iso_hom_inv {C : Type*} [Category C] (f : X ≅ Y) :
  f.hom ≫ f.inv = 𝟙 X := by simp [Iso.hom_inv_id]

@[simp] theorem iso_inv_hom {C : Type*} [Category C] (f : X ≅ Y) :
  f.inv ≫ f.hom = 𝟙 Y := by simp [Iso.inv_hom_id]

example (f : X ≅ Y) (g : Y ⟶ Z) :
  f.hom ≫ f.inv ≫ g = g := by simp

example (f : X ≅ Y) (g : W ⟶ X) :
  g ≫ f.hom ≫ f.inv = g := by simp

-- Complex categorical expressions
example (F : C ⥤ D) (G : D ⥤ E) (f : X ⟶ Y) (g : Y ⟶ Z) :
  (F ⋙ G).map (f ≫ 𝟙 Y ≫ g) = G.map (F.map f) ≫ G.map (F.map g) := by simp

example (F G H : C ⥤ D) (α : F ⟶ G) (β : G ⟶ H) (f : X ⟶ Y) :
  F.map f ≫ (α ≫ β).app Y = α.app X ≫ β.app X ≫ H.map f := by simp [NatTrans.comp_app]

-- Adjunction patterns (when available)
example (F : C ⥤ D) (G : D ⥤ C) (adj : F ⊣ G) (X : C) :
  adj.unit.app X ≫ G.map (F.map (𝟙 X)) = adj.unit.app X := by simp

-- Very frequent micro-patterns in category theory
example (f : X ⟶ Y) : f ≫ 𝟙 Y = f := by simp
example (f : X ⟶ Y) : 𝟙 X ≫ f = f := by simp
example (F : C ⥤ D) (X : C) : F.map (𝟙 X) = 𝟙 (F.obj X) := by simp
example (f : X ⟶ Y) : (𝟭 C).map f = f := by simp

-- Edge cases
example : (𝟙 X : X ⟶ X) ≫ 𝟙 X = 𝟙 X := by simp
example (F : C ⥤ D) : F.map (𝟙 X) ≫ 𝟙 (F.obj X) = 𝟙 (F.obj X) := by simp
example (f : X ≅ Y) : f.hom ≫ f.inv ≫ 𝟙 X = 𝟙 X := by simp