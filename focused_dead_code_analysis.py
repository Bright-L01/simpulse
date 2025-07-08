#!/usr/bin/env python3
"""
Focused Dead Code Analysis for Simpulse
Identifies the most impactful dead code that can be safely removed
"""

import re
from pathlib import Path
from typing import Dict, List, Set


def analyze_root_level_files():
    """Analyze Python files at the root level"""
    root_path = Path(".")
    root_files = []

    for file_path in root_path.glob("*.py"):
        if (
            file_path.name != "dead_code_analyzer.py"
            and file_path.name != "focused_dead_code_analysis.py"
        ):
            root_files.append(file_path)

    return root_files


def analyze_test_files():
    """Find standalone test files that don't correspond to real modules"""
    test_files = []

    # Find test files in root
    for file_path in Path(".").glob("test_*.py"):
        test_files.append(file_path)

    return test_files


def analyze_demo_files():
    """Find demo and debug files"""
    demo_files = []

    patterns = ["demo_*.py", "debug_*.py", "create_*.py", "run_*.py", "test_*.py"]

    for pattern in patterns:
        for file_path in Path(".").glob(pattern):
            demo_files.append(file_path)

    return demo_files


def check_imports_in_files(file_paths: List[Path]) -> Dict[str, Set[str]]:
    """Check what each file imports from simpulse"""
    imports = {}

    for file_path in file_paths:
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Find simpulse imports
            import_lines = re.findall(
                r"from\s+(?:src\.)?simpulse[.\w]*\s+import\s+[^#\n]+", content
            )
            simpulse_imports = set()

            for line in import_lines:
                simpulse_imports.add(line.strip())

            if simpulse_imports:
                imports[str(file_path)] = simpulse_imports

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return imports


def check_if_cli_integrated():
    """Check which CLI files are actually integrated into the main CLI"""
    cli_files = list(Path("src/simpulse").glob("cli_*.py"))

    # Check main CLI
    try:
        with open("src/simpulse/cli.py") as f:
            main_cli = f.read()

        integrated = []
        standalone = []

        for cli_file in cli_files:
            if cli_file.name == "cli.py":
                continue

            module_name = cli_file.stem
            if module_name in main_cli or f"from .{module_name}" in main_cli:
                integrated.append(cli_file)
            else:
                standalone.append(cli_file)

        return integrated, standalone

    except Exception as e:
        print(f"Error checking CLI integration: {e}")
        return [], cli_files


def analyze_core_module_usage():
    """Analyze which core modules are actually used"""
    core_modules = list(Path("src/simpulse").rglob("*.py"))

    # Check usage in main __init__.py
    try:
        with open("src/simpulse/__init__.py") as f:
            init_content = f.read()

        used_modules = set()
        unused_modules = []

        for module_path in core_modules:
            if module_path.name == "__init__.py":
                continue

            module_name = module_path.stem
            relative_path = module_path.relative_to(Path("src/simpulse"))

            # Check if imported in __init__.py
            if module_name in init_content or str(relative_path) in init_content:
                used_modules.add(str(relative_path))
            else:
                unused_modules.append(str(relative_path))

        return sorted(used_modules), sorted(unused_modules)

    except Exception as e:
        print(f"Error analyzing core module usage: {e}")
        return [], []


def find_large_unused_directories():
    """Find directories with many files that might be unused"""
    directories = {}

    for path in Path("src/simpulse").rglob("*.py"):
        if path.is_file():
            parent = path.parent
            if parent not in directories:
                directories[parent] = []
            directories[parent].append(path)

    # Sort by size
    large_dirs = []
    for dir_path, files in directories.items():
        if len(files) > 3:  # More than 3 files
            large_dirs.append((dir_path, len(files), files))

    return sorted(large_dirs, key=lambda x: x[1], reverse=True)


def main():
    print("=" * 80)
    print("FOCUSED DEAD CODE ANALYSIS - SIMPULSE")
    print("=" * 80)

    # 1. Root level files
    print("\n🗂️  ROOT LEVEL PYTHON FILES")
    root_files = analyze_root_level_files()
    print(f"Found {len(root_files)} Python files at root level:")
    for f in sorted(root_files)[:20]:  # Show first 20
        print(f"  {f}")
    if len(root_files) > 20:
        print(f"  ... and {len(root_files) - 20} more")

    # 2. Test files
    print("\n🧪 STANDALONE TEST FILES")
    test_files = analyze_test_files()
    print(f"Found {len(test_files)} standalone test files:")
    for f in sorted(test_files)[:15]:
        print(f"  {f}")
    if len(test_files) > 15:
        print(f"  ... and {len(test_files) - 15} more")

    # 3. Demo files
    print("\n🎬 DEMO/DEBUG FILES")
    demo_files = analyze_demo_files()
    print(f"Found {len(demo_files)} demo/debug files:")
    for f in sorted(demo_files)[:15]:
        print(f"  {f}")
    if len(demo_files) > 15:
        print(f"  ... and {len(demo_files) - 15} more")

    # 4. CLI integration
    print("\n⚙️  CLI INTEGRATION STATUS")
    integrated, standalone = check_if_cli_integrated()
    print(f"Integrated CLI modules: {len(integrated)}")
    for f in integrated:
        print(f"  ✅ {f}")
    print(f"Standalone CLI modules: {len(standalone)}")
    for f in standalone:
        print(f"  ❌ {f}")

    # 5. Core module usage
    print("\n🏗️  CORE MODULE USAGE")
    used, unused = analyze_core_module_usage()
    print(f"Used in __init__.py: {len(used)}")
    print(f"Not imported in __init__.py: {len(unused)}")

    # Show some unused modules
    print("\nSample unused modules:")
    for module in unused[:10]:
        print(f"  {module}")
    if len(unused) > 10:
        print(f"  ... and {len(unused) - 10} more")

    # 6. Large directories
    print("\n📁 LARGE DIRECTORIES")
    large_dirs = find_large_unused_directories()
    for dir_path, file_count, files in large_dirs[:5]:
        print(f"  {dir_path}: {file_count} files")

    # 7. Import analysis for root files
    print("\n📦 IMPORT ANALYSIS")
    all_files = root_files + test_files + demo_files
    imports = check_imports_in_files(all_files)

    print(f"Files importing from simpulse: {len(imports)}")
    for file_path, import_set in list(imports.items())[:10]:
        print(f"  {file_path}:")
        for imp in list(import_set)[:3]:
            print(f"    {imp}")
        if len(import_set) > 3:
            print(f"    ... and {len(import_set) - 3} more imports")

    # 8. RECOMMENDATIONS
    print("\n💡 REMOVAL RECOMMENDATIONS")
    print("Files that could likely be deleted with zero impact:")

    # Demo files are usually safe to remove
    safe_demo_files = [f for f in demo_files if "demo_" in f.name or "debug_" in f.name]
    print(f"\n1. Demo/Debug files ({len(safe_demo_files)}):")
    for f in safe_demo_files[:10]:
        print(f"   {f}")
    if len(safe_demo_files) > 10:
        print(f"   ... and {len(safe_demo_files) - 10} more")

    # Test files without corresponding modules
    print(f"\n2. Standalone test files ({len(test_files)}):")
    for f in test_files[:10]:
        print(f"   {f}")
    if len(test_files) > 10:
        print(f"   ... and {len(test_files) - 10} more")

    # CLI files not integrated
    print(f"\n3. Unintegrated CLI modules ({len(standalone)}):")
    for f in standalone:
        print(f"   {f}")

    print(f"\n📊 IMPACT SUMMARY")
    total_removable = len(safe_demo_files) + len(test_files) + len(standalone)
    print(f"Total files that could be removed: {total_removable}")
    print(f"Percentage of total Python files: {total_removable/263*100:.1f}%")


if __name__ == "__main__":
    main()
