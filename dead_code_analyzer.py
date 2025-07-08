#!/usr/bin/env python3
"""
Dead Code Analysis Tool for Simpulse
Systematically identifies unused code in the Python codebase
"""

import ast
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class DeadCodeAnalyzer:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.python_files = []
        self.classes = defaultdict(set)  # file -> set of class names
        self.functions = defaultdict(set)  # file -> set of function names
        self.imports = defaultdict(set)  # file -> set of imported names
        self.usage = defaultdict(set)  # name -> set of files using it
        self.variables = defaultdict(set)  # file -> set of variable names
        self.find_python_files()

    def find_python_files(self):
        """Find all Python files in the codebase"""
        for path in self.root_path.rglob("*.py"):
            if "__pycache__" not in str(path):
                self.python_files.append(path)
        print(f"Found {len(self.python_files)} Python files")

    def analyze_file(self, file_path: Path) -> Dict:
        """Analyze a single Python file"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            # Extract classes, functions, imports, and usage
            file_info = {
                "classes": set(),
                "functions": set(),
                "imports": set(),
                "variables": set(),
                "usage": set(),
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    file_info["classes"].add(node.name)
                elif isinstance(node, ast.FunctionDef):
                    file_info["functions"].add(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        file_info["imports"].add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            file_info["imports"].add(f"{node.module}.{alias.name}")
                elif isinstance(node, ast.Name):
                    file_info["usage"].add(node.id)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            file_info["variables"].add(target.id)

            return file_info

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return {}

    def analyze_all_files(self):
        """Analyze all Python files"""
        for file_path in self.python_files:
            rel_path = file_path.relative_to(self.root_path)
            file_info = self.analyze_file(file_path)

            if file_info:
                self.classes[str(rel_path)] = file_info["classes"]
                self.functions[str(rel_path)] = file_info["functions"]
                self.imports[str(rel_path)] = file_info["imports"]
                self.variables[str(rel_path)] = file_info["variables"]

                # Track usage
                for name in file_info["usage"]:
                    self.usage[name].add(str(rel_path))

    def find_unused_classes(self) -> List[Tuple[str, str]]:
        """Find classes that are defined but never used"""
        unused = []

        for file_path, classes in self.classes.items():
            for class_name in classes:
                # Check if class is used anywhere
                used = False
                for usage_file in self.usage.get(class_name, set()):
                    if usage_file != file_path:  # Don't count self-usage
                        used = True
                        break

                if not used:
                    unused.append((file_path, class_name))

        return unused

    def find_unused_functions(self) -> List[Tuple[str, str]]:
        """Find functions that are defined but never called"""
        unused = []

        for file_path, functions in self.functions.items():
            for func_name in functions:
                # Skip special methods and test functions
                if func_name.startswith("_") or func_name.startswith("test_"):
                    continue

                # Check if function is used anywhere
                used = False
                for usage_file in self.usage.get(func_name, set()):
                    if usage_file != file_path:  # Don't count self-usage
                        used = True
                        break

                if not used:
                    unused.append((file_path, func_name))

        return unused

    def find_unused_imports(self) -> List[Tuple[str, str]]:
        """Find imports that are never used"""
        unused = []

        for file_path, imports in self.imports.items():
            for import_name in imports:
                # Extract the actual name being imported
                name_parts = import_name.split(".")
                actual_name = name_parts[-1]

                # Check if imported name is used in the file
                if actual_name not in self.usage or file_path not in self.usage[actual_name]:
                    unused.append((file_path, import_name))

        return unused

    def find_files_with_no_references(self) -> List[str]:
        """Find Python files that are never imported"""
        referenced_files = set()

        # Find all files that are imported
        for file_path, imports in self.imports.items():
            for import_name in imports:
                # Convert import to potential file path
                import_path = import_name.replace(".", "/")
                for py_file in self.python_files:
                    if import_path in str(py_file):
                        referenced_files.add(str(py_file.relative_to(self.root_path)))

        # Find files that are never referenced
        unreferenced = []
        for file_path in self.python_files:
            rel_path = str(file_path.relative_to(self.root_path))
            if rel_path not in referenced_files:
                # Skip __init__.py and test files
                if not rel_path.endswith("__init__.py") and "test" not in rel_path:
                    unreferenced.append(rel_path)

        return unreferenced

    def find_duplicate_implementations(self) -> List[Tuple[str, str, str]]:
        """Find classes/functions with similar names indicating duplication"""
        duplicates = []

        # Group by similar names
        all_names = {}
        for file_path, classes in self.classes.items():
            for class_name in classes:
                if class_name not in all_names:
                    all_names[class_name] = []
                all_names[class_name].append((file_path, "class"))

        for file_path, functions in self.functions.items():
            for func_name in functions:
                if func_name not in all_names:
                    all_names[func_name] = []
                all_names[func_name].append((file_path, "function"))

        # Find duplicates
        for name, locations in all_names.items():
            if len(locations) > 1:
                for file_path, item_type in locations:
                    duplicates.append((name, file_path, item_type))

        return duplicates

    def generate_report(self):
        """Generate comprehensive dead code report"""
        print("=" * 80)
        print("DEAD CODE ANALYSIS REPORT")
        print("=" * 80)

        unused_classes = self.find_unused_classes()
        unused_functions = self.find_unused_functions()
        unused_imports = self.find_unused_imports()
        unreferenced_files = self.find_files_with_no_references()
        duplicates = self.find_duplicate_implementations()

        print(f"\n📊 SUMMARY")
        print(f"Total Python files: {len(self.python_files)}")
        print(f"Unused classes: {len(unused_classes)}")
        print(f"Unused functions: {len(unused_functions)}")
        print(f"Unused imports: {len(unused_imports)}")
        print(f"Unreferenced files: {len(unreferenced_files)}")
        print(f"Potential duplicates: {len(duplicates)}")

        if unused_classes:
            print(f"\n🏗️  UNUSED CLASSES ({len(unused_classes)})")
            for file_path, class_name in sorted(unused_classes):
                print(f"  {file_path}:{class_name}")

        if unused_functions:
            print(f"\n🔧 UNUSED FUNCTIONS ({len(unused_functions)})")
            for file_path, func_name in sorted(unused_functions):
                print(f"  {file_path}:{func_name}")

        if unused_imports:
            print(f"\n📦 UNUSED IMPORTS ({len(unused_imports)})")
            for file_path, import_name in sorted(unused_imports):
                print(f"  {file_path}:{import_name}")

        if unreferenced_files:
            print(f"\n📄 UNREFERENCED FILES ({len(unreferenced_files)})")
            for file_path in sorted(unreferenced_files):
                print(f"  {file_path}")

        if duplicates:
            print(f"\n🔄 POTENTIAL DUPLICATES ({len(duplicates)})")
            duplicate_groups = defaultdict(list)
            for name, file_path, item_type in duplicates:
                duplicate_groups[name].append((file_path, item_type))

            for name, locations in duplicate_groups.items():
                if len(locations) > 1:
                    print(f"  {name}:")
                    for file_path, item_type in locations:
                        print(f"    - {file_path} ({item_type})")


def main():
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
    else:
        root_path = "."

    analyzer = DeadCodeAnalyzer(root_path)
    analyzer.analyze_all_files()
    analyzer.generate_report()


if __name__ == "__main__":
    main()
