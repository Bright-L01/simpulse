-- Complex pattern matching rules
namespace Complex

@[simp] theorem complex_pattern_0 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_0 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
@[simp] theorem complex_pattern_1 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_1 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
@[simp] theorem complex_pattern_2 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_2 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
@[simp] theorem complex_pattern_3 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_3 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
@[simp] theorem complex_pattern_4 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_4 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
@[simp] theorem complex_pattern_5 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_5 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
@[simp] theorem complex_pattern_6 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_6 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
@[simp] theorem complex_pattern_7 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_7 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
@[simp] theorem complex_pattern_8 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_8 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
@[simp] theorem complex_pattern_9 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_9 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
@[simp] theorem complex_pattern_10 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_10 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
@[simp] theorem complex_pattern_11 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_11 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
@[simp] theorem complex_pattern_12 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_12 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
@[simp] theorem complex_pattern_13 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_13 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
@[simp] theorem complex_pattern_14 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_14 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
@[simp] theorem complex_pattern_15 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_15 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
@[simp] theorem complex_pattern_16 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_16 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
@[simp] theorem complex_pattern_17 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_17 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
@[simp] theorem complex_pattern_18 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_18 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl
@[simp] theorem complex_pattern_19 (a b c d : Nat) (h1 : a < b) (h2 : b < c) : 
  (a * b + c * d) * (a + b + c + d) = (a * b + c * d) * (a + b + c + d) := rfl

@[simp] theorem nested_match_19 (x y z : Nat) :
  match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y  
  | x, 0, 0 => x
  | x, y, z => x + y + z
  = match x, y, z with
  | 0, 0, z => z
  | 0, y, 0 => y
  | x, 0, 0 => x
  | x, y, z => x + y + z := rfl

end Complex
