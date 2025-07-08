-- StringOperations.lean
-- String manipulation with heavy simp usage
-- Expected: 2x+ speedup from frequent string simplification rules

-- Core string rules (very frequently used)
@[simp] theorem string_append_empty (s : String) : s ++ "" = s := by
  rfl

@[simp] theorem empty_append_string (s : String) : "" ++ s = s := by
  rfl

@[simp] theorem string_length_empty : "".length = 0 := by
  rfl

@[simp] theorem string_append_assoc (s1 s2 s3 : String) : 
  (s1 ++ s2) ++ s3 = s1 ++ (s2 ++ s3) := by
  rfl

-- Character operations (frequently used)
@[simp] theorem string_push_empty (c : Char) : String.push "" c = String.singleton c := by
  rfl

@[simp] theorem string_singleton_length (c : Char) : (String.singleton c).length = 1 := by
  rfl

-- List conversion rules (moderately used)
@[simp] theorem string_to_list_empty : "".toList = [] := by
  rfl

@[simp] theorem string_of_list_nil : String.ofList [] = "" := by
  rfl

@[simp] theorem string_of_list_to_list (s : String) : 
  String.ofList s.toList = s := by
  rfl

-- Heavy usage patterns (simulates real string processing)
example (s1 s2 s3 : String) : 
  (s1 ++ s2) ++ s3 ++ "" = s1 ++ s2 ++ s3 := by simp

example (s : String) : 
  s ++ "" ++ "" = s := by simp

example (s1 s2 : String) :
  "" ++ s1 ++ "" ++ s2 = s1 ++ s2 := by simp

example (s1 s2 s3 s4 : String) :
  ((s1 ++ s2) ++ "") ++ ((s3 ++ "") ++ s4) = s1 ++ s2 ++ s3 ++ s4 := by simp

-- String building patterns (very common)
example (s : String) (c1 c2 : Char) :
  s ++ String.singleton c1 ++ String.singleton c2 ++ "" = 
  s ++ String.singleton c1 ++ String.singleton c2 := by simp

example (s1 s2 : String) :
  (s1 ++ "") ++ (s2 ++ "") = s1 ++ s2 := by simp

-- Conversion round-trips
example (chars : List Char) :
  (String.ofList chars).toList = chars := by
  simp [String.toList_ofList]

example (s : String) :
  String.ofList (s.toList ++ []) = s := by simp

-- Nested operations
example (s1 s2 s3 : String) :
  ((s1 ++ "") ++ (s2 ++ "")) ++ (s3 ++ "") = s1 ++ s2 ++ s3 := by simp

example (s : String) (c : Char) :
  (s ++ "").push c = s.push c := by simp [String.push_append]

-- Length calculations (frequent in real code)
example (s1 s2 : String) :
  (s1 ++ s2 ++ "").length = s1.length + s2.length := by simp

example (s : String) :
  (s ++ "").length = s.length := by simp

example (c : Char) :
  (String.singleton c ++ "").length = 1 := by simp

-- Character manipulation
example (s : String) (c1 c2 c3 : Char) :
  s ++ String.singleton c1 ++ String.singleton c2 ++ String.singleton c3 ++ "" =
  s ++ String.singleton c1 ++ String.singleton c2 ++ String.singleton c3 := by simp

-- Prefix/suffix patterns
example (prefix suffix content : String) :
  prefix ++ content ++ suffix ++ "" = prefix ++ content ++ suffix := by simp

example (s1 s2 s3 : String) :
  "" ++ s1 ++ "" ++ s2 ++ "" ++ s3 ++ "" = s1 ++ s2 ++ s3 := by simp

-- Format string patterns (common in real applications)
example (name : String) (value : String) :
  name ++ "=" ++ value ++ "" = name ++ "=" ++ value := by simp

example (tag : String) (content : String) :
  "<" ++ tag ++ ">" ++ content ++ "</" ++ tag ++ ">" ++ "" =
  "<" ++ tag ++ ">" ++ content ++ "</" ++ tag ++ ">" := by simp

-- Empty string elimination in complex expressions
example (s1 s2 s3 s4 s5 : String) :
  s1 ++ "" ++ s2 ++ "" ++ s3 ++ "" ++ s4 ++ "" ++ s5 ++ "" =
  s1 ++ s2 ++ s3 ++ s4 ++ s5 := by simp

-- Associativity with empty strings
example (a b c d : String) :
  ((a ++ "") ++ (b ++ "")) ++ ((c ++ "") ++ (d ++ "")) =
  a ++ b ++ c ++ d := by simp

-- Character sequence building
example (c1 c2 c3 c4 : Char) :
  String.singleton c1 ++ String.singleton c2 ++ String.singleton c3 ++ String.singleton c4 ++ "" =
  String.singleton c1 ++ String.singleton c2 ++ String.singleton c3 ++ String.singleton c4 := by simp

-- Template processing patterns
example (header body footer : String) :
  (header ++ "") ++ (body ++ "") ++ (footer ++ "") = header ++ body ++ footer := by simp

-- URL/path construction patterns
example (protocol domain path : String) :
  protocol ++ "://" ++ domain ++ "/" ++ path ++ "" = 
  protocol ++ "://" ++ domain ++ "/" ++ path := by simp

-- Very frequent micro-patterns
example (s : String) : s ++ "" = s := by simp
example (s : String) : "" ++ s = s := by simp
example (s1 s2 : String) : (s1 ++ "") ++ s2 = s1 ++ s2 := by simp
example (s1 s2 : String) : s1 ++ ("" ++ s2) = s1 ++ s2 := by simp

-- Edge cases
example : "" ++ "" = "" := by simp
example : ("" ++ "").length = 0 := by simp
example (c : Char) : String.singleton c ++ "" ++ "" = String.singleton c := by simp