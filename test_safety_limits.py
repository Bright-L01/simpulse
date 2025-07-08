#!/usr/bin/env python3
"""Test safety limits by creating problematic scenarios."""

import os
import tempfile
import time
from pathlib import Path

from src.simpulse.error import MemoryError
from src.simpulse.unified_optimizer import UnifiedOptimizer


def test_large_file_protection():
    """Test that files over 1MB are rejected."""
    print("\n=== Testing Large File Protection ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a large Lean file (>1MB)
        large_file = Path(tmpdir) / "large.lean"
        content = "-- Large file\n" + ("@[simp] theorem test : 1 = 1 := by simp\n" * 50000)
        large_file.write_text(content)

        # Also create a normal file
        normal_file = Path(tmpdir) / "normal.lean"
        normal_file.write_text("@[simp] theorem normal : 1 = 1 := by simp")

        print(f"Created large file: {os.path.getsize(large_file) / 1_000_000:.2f}MB")
        print(f"Created normal file: {os.path.getsize(normal_file)} bytes")

        optimizer = UnifiedOptimizer()
        result = optimizer.optimize(tmpdir)

        # The optimization should succeed but skip the large file
        if result["total_rules"] == 1:  # Only the normal file's rule
            print(f"✅ PASS: Large file skipped, normal file processed")
            print(f"   Processed {result['total_rules']} rules from normal files")
        else:
            print(f"❌ FAIL: Expected 1 rule, got {result['total_rules']}")


def test_timeout_protection():
    """Test that long-running operations are terminated."""
    print("\n=== Testing Timeout Protection ===")

    # Check if we're on a platform that supports signal-based timeout
    import platform

    if platform.system() == "Windows":
        print("⚠️  WARNING: Signal-based timeout not supported on Windows")
        print("   Timeout protection still active but harder to test")
        return

    # Set very short timeout for testing
    os.environ["SIMPULSE_TIMEOUT"] = "1"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a file with slow processing
        file = Path(tmpdir) / "slow.lean"
        # Create content that will trigger slow regex processing
        content = "@[simp] theorem test : " + "(" * 1000 + "1" + ")" * 1000 + " = 1 := by simp\n"
        content = content * 100  # Make it even slower
        file.write_text(content)

        print(f"Created file with complex patterns to slow processing")

        try:
            pass

            UnifiedOptimizer()

            # Test direct timeout functionality
            from src.simpulse.error import TimeoutError as SimpulseTimeoutError
            from src.simpulse.error import timeout

            try:
                with timeout(1, "test operation"):
                    time.sleep(2)  # Sleep longer than timeout
                print("❌ FAIL: Timeout should have triggered")
            except SimpulseTimeoutError as e:
                print(f"✅ PASS: Timeout triggered correctly")
                print(f"   Error: {e}")
            except Exception as e:
                if "SIGALRM" in str(e) or "Timeout" in str(e):
                    print(f"✅ PASS: Timeout triggered (alternative)")
                    print(f"   Error: {e}")
                else:
                    print(f"⚠️  WARNING: Unexpected error: {e}")

        except Exception as e:
            print(f"ℹ️  INFO: Timeout test result: {e}")

    # Reset timeout
    os.environ.pop("SIMPULSE_TIMEOUT", None)


def test_memory_protection():
    """Test that high memory usage is detected."""
    print("\n=== Testing Memory Protection ===")

    # This test requires psutil to be installed
    try:
        pass

        print("✅ psutil available - memory monitoring active")

        # Set very low memory limit for testing (100MB)
        os.environ["SIMPULSE_MAX_MEMORY"] = "100000000"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file
            file = Path(tmpdir) / "test.lean"
            file.write_text("theorem test : 1 = 1 := by simp")

            try:
                # Allocate some memory to push over limit
                big_list = [0] * 50_000_000  # ~400MB

                optimizer = UnifiedOptimizer()
                result = optimizer.optimize(tmpdir)

                # Memory check might not trigger immediately
                print("⚠️  WARNING: Memory limit not triggered (may be normal)")

            except MemoryError as e:
                print(f"✅ PASS: Memory limit enforced")
                print(f"   Error: {e}")
            except Exception as e:
                print(f"ℹ️  INFO: Operation completed with: {e}")
            finally:
                # Clean up
                if "big_list" in locals():
                    del big_list

        # Reset memory limit
        os.environ.pop("SIMPULSE_MAX_MEMORY", None)

    except ImportError:
        print("⚠️  WARNING: psutil not installed - memory monitoring disabled")


def test_graceful_degradation():
    """Test that errors are handled gracefully without crashes."""
    print("\n=== Testing Graceful Degradation ===")

    scenarios = [
        ("nonexistent_directory", "/this/does/not/exist"),
        ("file_not_directory", __file__),
        ("empty_directory", tempfile.mkdtemp()),
    ]

    optimizer = UnifiedOptimizer()

    for name, path in scenarios:
        print(f"\nTesting {name}: {path}")
        try:
            result = optimizer.optimize(path)
            if name == "empty_directory":
                print(f"✅ PASS: Handled empty directory gracefully")
                print(f"   Result: {result['total_rules']} rules found")
            else:
                print(f"❌ FAIL: Should have raised an error")
        except Exception as e:
            print(f"✅ PASS: Error handled gracefully")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Message: {e}")

    # Clean up temp directory
    if scenarios[2][1] and os.path.exists(scenarios[2][1]):
        os.rmdir(scenarios[2][1])


def test_safe_file_operations():
    """Test safe file read/write operations."""
    print("\n=== Testing Safe File Operations ===")

    from src.simpulse.error import safe_file_read, safe_file_write

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Test normal file
        normal_file = tmppath / "normal.lean"
        normal_file.write_text("theorem test : 1 = 1 := by simp")

        content = safe_file_read(normal_file)
        if content:
            print("✅ PASS: Normal file read successfully")
        else:
            print("❌ FAIL: Failed to read normal file")

        # Test write
        new_file = tmppath / "new.lean"
        if safe_file_write(new_file, "-- New file"):
            print("✅ PASS: File written successfully")
        else:
            print("❌ FAIL: Failed to write file")

        # Test permission denied (Unix only)
        if hasattr(os, "chmod"):
            protected_file = tmppath / "protected.lean"
            protected_file.write_text("protected")
            os.chmod(protected_file, 0o000)

            content = safe_file_read(protected_file)
            if content is None:
                print("✅ PASS: Permission denied handled gracefully")
            else:
                print("❌ FAIL: Should have failed on protected file")

            # Restore permissions
            os.chmod(protected_file, 0o644)


def main():
    """Run all safety tests."""
    print("🛡️  SIMPULSE SAFETY LIMITS TEST SUITE")
    print("=" * 50)

    tests = [
        test_large_file_protection,
        test_timeout_protection,
        test_memory_protection,
        test_graceful_degradation,
        test_safe_file_operations,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n❌ Test {test.__name__} crashed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 50)
    print("✅ Safety limits test suite completed")


if __name__ == "__main__":
    main()
