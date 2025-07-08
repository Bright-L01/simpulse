#!/usr/bin/env python3
"""Test optimizer with problematic files from code audit."""

import os
import tempfile
from pathlib import Path

from src.simpulse.error import safe_file_read
from src.simpulse.unified_optimizer import UnifiedOptimizer


def create_problematic_files(tmpdir):
    """Create various problematic files that caused issues in the past."""
    files = []

    # 1. File with Unicode characters
    unicode_file = Path(tmpdir) / "unicode.lean"
    content = """
-- File with unicode: 数学 𝔽 ∀ ∃ ∈ ∉
@[simp] theorem unicode_test : ∀ x : ℕ, x = x := by simp
"""
    unicode_file.write_text(content, encoding="utf-8")
    files.append(("Unicode characters", unicode_file))

    # 2. File with malformed simp rules
    malformed_file = Path(tmpdir) / "malformed.lean"
    content = """
@[simp malformed syntax
theorem bad1 : 1 = 1 := by simp

@[simp] @[simp] -- duplicate attribute
theorem bad2 : 2 = 2 := by simp

@[simp]
-- missing theorem
"""
    malformed_file.write_text(content)
    files.append(("Malformed syntax", malformed_file))

    # 3. File with deeply nested expressions
    nested_file = Path(tmpdir) / "nested.lean"
    content = "@[simp] theorem nested : "
    for i in range(50):
        content += f"({i} + "
    content += "0"
    for i in range(50):
        content += ")"
    content += " = 1225 := by simp\n"
    nested_file.write_text(content)
    files.append(("Deeply nested", nested_file))

    # 4. File with very long lines
    longline_file = Path(tmpdir) / "longline.lean"
    content = f"@[simp] theorem longline : {'1 + ' * 500}1 = 501 := by simp\n"
    longline_file.write_text(content)
    files.append(("Very long lines", longline_file))

    # 5. Empty file
    empty_file = Path(tmpdir) / "empty.lean"
    empty_file.write_text("")
    files.append(("Empty file", empty_file))

    # 6. File with only comments
    comment_file = Path(tmpdir) / "comments.lean"
    content = """
-- This file has only comments
-- No actual Lean code
/-
  Multi-line comment
  Still no code
-/
"""
    comment_file.write_text(content)
    files.append(("Comments only", comment_file))

    # 7. File with special characters in rules
    special_file = Path(tmpdir) / "special.lean"
    content = """
@[simp] theorem rule'with'quotes : 1 = 1 := by simp
@[simp] theorem rule_with-dash : 2 = 2 := by simp
@[simp] theorem rule.with.dots : 3 = 3 := by simp
"""
    special_file.write_text(content)
    files.append(("Special characters", special_file))

    # 8. Binary file (should be skipped)
    binary_file = Path(tmpdir) / "binary.lean"
    binary_file.write_bytes(b"\x00\x01\x02\x03\x04\x05\x06\x07")
    files.append(("Binary file", binary_file))

    return files


def test_problematic_files():
    """Test optimizer handles problematic files gracefully."""
    print("\n🔍 TESTING PROBLEMATIC FILES FROM AUDIT")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create all problematic files
        test_files = create_problematic_files(tmpdir)

        # Also create one normal file to ensure processing continues
        normal_file = Path(tmpdir) / "normal.lean"
        normal_file.write_text(
            """
@[simp] theorem normal1 : 1 = 1 := by simp
@[simp] theorem normal2 : 2 = 2 := by simp

example : 1 = 1 := by simp [normal1]
example : 2 = 2 := by simp [normal2]
"""
        )

        print(f"Created {len(test_files)} problematic files + 1 normal file")
        print("\nTesting individual file handling:")

        # Test each problematic file individually
        for desc, file in test_files:
            print(f"\n📄 {desc}: {file.name}")
            content = safe_file_read(file)
            if content is not None:
                print(f"   ✅ File read successfully ({len(content)} chars)")
            else:
                print(f"   ⚠️  File could not be read safely")

        # Test full optimization with all files
        print("\n\n🔧 RUNNING FULL OPTIMIZATION")
        print("-" * 30)

        optimizer = UnifiedOptimizer()

        try:
            result = optimizer.optimize(tmpdir)

            print(f"\n✅ Optimization completed successfully!")
            print(f"   Total rules found: {result['total_rules']}")
            print(f"   Rules changed: {result['rules_changed']}")
            print(f"   Estimated improvement: {result['estimated_improvement']}%")

            if result["total_rules"] >= 2:  # At least the normal file's rules
                print(f"\n✅ PASS: Problematic files handled gracefully")
                print(f"   Normal processing continued despite issues")
            else:
                print(f"\n❌ FAIL: Expected at least 2 rules from normal file")

        except Exception as e:
            print(f"\n❌ FAIL: Optimization crashed!")
            print(f"   Error: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()


def test_edge_cases():
    """Test additional edge cases."""
    print("\n\n🔬 TESTING ADDITIONAL EDGE CASES")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Symlink (if supported)
        if hasattr(os, "symlink"):
            try:
                target = Path(tmpdir) / "target.lean"
                target.write_text("@[simp] theorem target : 1 = 1 := by simp")

                symlink = Path(tmpdir) / "symlink.lean"
                symlink.symlink_to(target)

                print("\n📎 Testing symlink handling...")
                content = safe_file_read(symlink)
                if content:
                    print("   ✅ Symlinks handled correctly")
                else:
                    print("   ⚠️  Symlink not readable")
            except Exception as e:
                print(f"   ℹ️  Symlink test skipped: {e}")

        # 2. Directory with spaces and special chars
        weird_dir = Path(tmpdir) / "my project (v2.0) - test!"
        weird_dir.mkdir()

        weird_file = weird_dir / "file with spaces.lean"
        weird_file.write_text("@[simp] theorem spaces : 1 = 1 := by simp")

        print("\n📁 Testing paths with special characters...")
        optimizer = UnifiedOptimizer()

        try:
            result = optimizer.optimize(weird_dir)
            if result["total_rules"] > 0:
                print("   ✅ Special characters in paths handled")
            else:
                print("   ❌ Failed to process files in special path")
        except Exception as e:
            print(f"   ❌ Error with special paths: {e}")

        # 3. Read-only directory (Unix only)
        if hasattr(os, "chmod"):
            readonly_dir = Path(tmpdir) / "readonly"
            readonly_dir.mkdir()

            file = readonly_dir / "test.lean"
            file.write_text("@[simp] theorem ro : 1 = 1 := by simp")

            # Make directory read-only
            os.chmod(readonly_dir, 0o555)

            print("\n🔒 Testing read-only directory...")
            result = optimizer.optimize(readonly_dir)

            if result["total_rules"] > 0:
                print("   ✅ Read-only files processed correctly")
            else:
                print("   ⚠️  Could not process read-only files")

            # Restore permissions for cleanup
            os.chmod(readonly_dir, 0o755)


def main():
    """Run all problematic file tests."""
    test_problematic_files()
    test_edge_cases()

    print("\n\n" + "=" * 50)
    print("✅ Problematic files test suite completed")
    print("\nSUMMARY: The optimizer should handle all edge cases gracefully")
    print("without crashing, continuing to process valid files.")


if __name__ == "__main__":
    main()
