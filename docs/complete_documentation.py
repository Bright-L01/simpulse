#!/usr/bin/env python3
"""
Documentation consolidation tool for Simpulse.

This script generates comprehensive documentation by consolidating
all project documentation, code docstrings, and examples.
"""

import argparse
import ast
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModuleDoc:
    """Documentation for a Python module."""
    name: str
    path: Path
    docstring: Optional[str]
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    imports: List[str]
    examples: List[str]


@dataclass
class APIDoc:
    """API documentation entry."""
    name: str
    type: str  # class, function, method
    signature: str
    docstring: Optional[str]
    parameters: List[Dict[str, str]]
    returns: Optional[str]
    examples: List[str]
    source_file: str
    line_number: int


class DocumentationGenerator:
    """Generates comprehensive documentation for Simpulse."""
    
    def __init__(self, project_root: Path):
        """Initialize documentation generator.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.src_dir = project_root / "src" / "simpulse"
        self.docs_dir = project_root / "docs"
        self.examples_dir = project_root / "examples"
        
    def extract_module_docs(self, module_path: Path) -> Optional[ModuleDoc]:
        """Extract documentation from a Python module.
        
        Args:
            module_path: Path to Python module
            
        Returns:
            ModuleDoc object or None if extraction fails
        """
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(module_path))
            
            # Extract module docstring
            module_docstring = ast.get_docstring(tree)
            
            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
            
            # Extract classes and functions
            classes = []
            functions = []
            examples = []
            
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_doc = self._extract_class_doc(node, module_path)
                    if class_doc:
                        classes.append(class_doc)
                        
                elif isinstance(node, ast.FunctionDef):
                    func_doc = self._extract_function_doc(node, module_path)
                    if func_doc:
                        functions.append(func_doc)
            
            # Extract examples from docstrings
            if module_docstring:
                examples.extend(self._extract_examples(module_docstring))
            
            # Calculate module name
            relative_path = module_path.relative_to(self.src_dir)
            module_name = str(relative_path.with_suffix('')).replace('/', '.')
            
            return ModuleDoc(
                name=module_name,
                path=module_path,
                docstring=module_docstring,
                classes=classes,
                functions=functions,
                imports=imports,
                examples=examples
            )
            
        except Exception as e:
            logger.error(f"Failed to extract docs from {module_path}: {e}")
            return None
    
    def _extract_class_doc(self, node: ast.ClassDef, file_path: Path) -> Dict[str, Any]:
        """Extract documentation from a class definition.
        
        Args:
            node: AST ClassDef node
            file_path: Source file path
            
        Returns:
            Class documentation dictionary
        """
        class_doc = {
            "name": node.name,
            "line": node.lineno,
            "docstring": ast.get_docstring(node),
            "methods": [],
            "attributes": [],
            "bases": [self._get_name(base) for base in node.bases]
        }
        
        # Extract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_doc = self._extract_function_doc(item, file_path)
                if method_doc:
                    class_doc["methods"].append(method_doc)
        
        return class_doc
    
    def _extract_function_doc(self, node: ast.FunctionDef, file_path: Path) -> Dict[str, Any]:
        """Extract documentation from a function definition.
        
        Args:
            node: AST FunctionDef node
            file_path: Source file path
            
        Returns:
            Function documentation dictionary
        """
        # Extract parameters
        params = []
        for arg in node.args.args:
            param = {"name": arg.arg}
            
            # Get type annotation
            if arg.annotation:
                param["type"] = self._get_annotation(arg.annotation)
            
            params.append(param)
        
        # Extract return type
        return_type = None
        if node.returns:
            return_type = self._get_annotation(node.returns)
        
        # Parse docstring for additional info
        docstring = ast.get_docstring(node)
        param_docs = {}
        return_doc = None
        examples = []
        
        if docstring:
            param_docs, return_doc = self._parse_docstring(docstring)
            examples = self._extract_examples(docstring)
        
        # Merge parameter documentation
        for param in params:
            if param["name"] in param_docs:
                param["description"] = param_docs[param["name"]]
        
        return {
            "name": node.name,
            "line": node.lineno,
            "docstring": docstring,
            "parameters": params,
            "return_type": return_type,
            "return_description": return_doc,
            "examples": examples,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "decorators": [self._get_name(d) for d in node.decorator_list]
        }
    
    def _get_name(self, node: ast.AST) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        else:
            return ast.dump(node)
    
    def _get_annotation(self, node: ast.AST) -> str:
        """Get type annotation as string."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[{self._get_annotation(node.slice)}]"
        elif isinstance(node, ast.Tuple):
            elements = [self._get_annotation(e) for e in node.elts]
            return f"Tuple[{', '.join(elements)}]"
        elif isinstance(node, ast.List):
            elements = [self._get_annotation(e) for e in node.elts]
            return f"List[{', '.join(elements)}]"
        else:
            return ast.dump(node)
    
    def _parse_docstring(self, docstring: str) -> Tuple[Dict[str, str], Optional[str]]:
        """Parse docstring to extract parameter and return documentation.
        
        Args:
            docstring: Docstring text
            
        Returns:
            Tuple of (param_docs, return_doc)
        """
        param_docs = {}
        return_doc = None
        
        # Simple regex-based parsing
        # Args section
        args_match = re.search(r'Args?:\s*\n((?:\s+\w+.*\n)+)', docstring, re.MULTILINE)
        if args_match:
            args_text = args_match.group(1)
            # Parse individual parameters
            param_pattern = re.compile(r'^\s+(\w+):\s*(.+)$', re.MULTILINE)
            for match in param_pattern.finditer(args_text):
                param_name = match.group(1)
                param_desc = match.group(2).strip()
                param_docs[param_name] = param_desc
        
        # Returns section
        returns_match = re.search(r'Returns?:\s*\n\s+(.+?)(?:\n\n|\n\s*\w+:|\Z)', docstring, re.MULTILINE | re.DOTALL)
        if returns_match:
            return_doc = returns_match.group(1).strip()
        
        return param_docs, return_doc
    
    def _extract_examples(self, text: str) -> List[str]:
        """Extract code examples from text.
        
        Args:
            text: Text containing examples
            
        Returns:
            List of example code blocks
        """
        examples = []
        
        # Look for code blocks
        code_block_pattern = re.compile(r'```python\n(.*?)\n```', re.DOTALL)
        for match in code_block_pattern.finditer(text):
            examples.append(match.group(1))
        
        # Look for doctest examples
        doctest_pattern = re.compile(r'>>>\s*(.+?)(?=\n>>>|\n\n|\Z)', re.DOTALL)
        for match in doctest_pattern.finditer(text):
            examples.append(match.group(1))
        
        return examples
    
    def generate_api_reference(self) -> List[APIDoc]:
        """Generate complete API reference.
        
        Returns:
            List of API documentation entries
        """
        api_docs = []
        
        # Walk through all Python files
        for py_file in self.src_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            module_doc = self.extract_module_docs(py_file)
            if not module_doc:
                continue
            
            # Add module-level functions
            for func in module_doc.functions:
                api_doc = APIDoc(
                    name=f"{module_doc.name}.{func['name']}",
                    type="function",
                    signature=self._build_signature(func),
                    docstring=func.get("docstring"),
                    parameters=func.get("parameters", []),
                    returns=func.get("return_type"),
                    examples=func.get("examples", []),
                    source_file=str(py_file.relative_to(self.project_root)),
                    line_number=func.get("line", 0)
                )
                api_docs.append(api_doc)
            
            # Add classes and methods
            for cls in module_doc.classes:
                # Add class itself
                class_api_doc = APIDoc(
                    name=f"{module_doc.name}.{cls['name']}",
                    type="class",
                    signature=cls['name'],
                    docstring=cls.get("docstring"),
                    parameters=[],
                    returns=None,
                    examples=[],
                    source_file=str(py_file.relative_to(self.project_root)),
                    line_number=cls.get("line", 0)
                )
                api_docs.append(class_api_doc)
                
                # Add methods
                for method in cls.get("methods", []):
                    method_api_doc = APIDoc(
                        name=f"{module_doc.name}.{cls['name']}.{method['name']}",
                        type="method",
                        signature=self._build_signature(method),
                        docstring=method.get("docstring"),
                        parameters=method.get("parameters", []),
                        returns=method.get("return_type"),
                        examples=method.get("examples", []),
                        source_file=str(py_file.relative_to(self.project_root)),
                        line_number=method.get("line", 0)
                    )
                    api_docs.append(method_api_doc)
        
        return api_docs
    
    def _build_signature(self, func_doc: Dict[str, Any]) -> str:
        """Build function signature string.
        
        Args:
            func_doc: Function documentation dictionary
            
        Returns:
            Signature string
        """
        params = []
        for param in func_doc.get("parameters", []):
            param_str = param["name"]
            if "type" in param:
                param_str += f": {param['type']}"
            params.append(param_str)
        
        signature = f"{func_doc['name']}({', '.join(params)})"
        
        if func_doc.get("return_type"):
            signature += f" -> {func_doc['return_type']}"
            
        return signature
    
    def generate_markdown_docs(self, output_dir: Path) -> bool:
        """Generate Markdown documentation files.
        
        Args:
            output_dir: Directory to write documentation
            
        Returns:
            True if successful
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate API reference
            api_docs = self.generate_api_reference()
            
            # Group by module
            by_module = {}
            for api_doc in api_docs:
                module = api_doc.name.rsplit('.', 2)[0]
                if module not in by_module:
                    by_module[module] = []
                by_module[module].append(api_doc)
            
            # Generate index
            index_path = output_dir / "index.md"
            self._generate_index(index_path, by_module)
            
            # Generate module pages
            for module, docs in by_module.items():
                module_file = output_dir / f"{module}.md"
                self._generate_module_page(module_file, module, docs)
            
            # Generate full API reference
            api_ref_path = output_dir / "api_reference.md"
            self._generate_api_reference_page(api_ref_path, api_docs)
            
            # Generate examples page
            examples_path = output_dir / "examples.md"
            self._generate_examples_page(examples_path)
            
            logger.info(f"Documentation generated in {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate documentation: {e}")
            return False
    
    def _generate_index(self, output_path: Path, modules: Dict[str, List[APIDoc]]):
        """Generate index page."""
        lines = [
            "# Simpulse Documentation",
            "",
            "Welcome to the Simpulse documentation!",
            "",
            "## 🚀 Quick Start",
            "",
            "```python",
            "from simpulse import Simpulse",
            "",
            "# Initialize Simpulse",
            'simpulse = Simpulse(project_path=".")',
            "",
            "# Run optimization",
            'result = await simpulse.optimize(modules=["MyModule"])',
            "```",
            "",
            "## 📚 Modules",
            ""
        ]
        
        # List modules
        for module in sorted(modules.keys()):
            lines.append(f"- [{module}]({module}.md)")
        
        lines.extend([
            "",
            "## 📖 Additional Resources",
            "",
            "- [API Reference](api_reference.md)",
            "- [Examples](examples.md)",
            "- [GitHub Repository](https://github.com/your-org/simpulse)",
            "",
            f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        ])
        
        output_path.write_text('\n'.join(lines))
    
    def _generate_module_page(self, output_path: Path, module: str, docs: List[APIDoc]):
        """Generate documentation page for a module."""
        lines = [
            f"# {module}",
            ""
        ]
        
        # Group by type
        classes = [d for d in docs if d.type == "class"]
        functions = [d for d in docs if d.type == "function"]
        
        # Module overview
        module_parts = module.split('.')
        if module_parts:
            lines.extend([
                "## Overview",
                "",
                f"Module: `{module}`",
                ""
            ])
        
        # Classes
        if classes:
            lines.extend(["## Classes", ""])
            for cls in sorted(classes, key=lambda x: x.name):
                lines.append(f"### {cls.name.split('.')[-1]}")
                lines.append("")
                if cls.docstring:
                    lines.append(cls.docstring)
                    lines.append("")
                
                # List methods
                methods = [d for d in docs if d.type == "method" and d.name.startswith(cls.name + ".")]
                if methods:
                    lines.append("**Methods:**")
                    lines.append("")
                    for method in methods:
                        method_name = method.name.split('.')[-1]
                        lines.append(f"- `{method.signature}`")
                    lines.append("")
        
        # Functions
        if functions:
            lines.extend(["## Functions", ""])
            for func in sorted(functions, key=lambda x: x.name):
                lines.append(f"### {func.name.split('.')[-1]}")
                lines.append("")
                lines.append(f"```python")
                lines.append(func.signature)
                lines.append("```")
                lines.append("")
                if func.docstring:
                    lines.append(func.docstring)
                    lines.append("")
        
        output_path.write_text('\n'.join(lines))
    
    def _generate_api_reference_page(self, output_path: Path, api_docs: List[APIDoc]):
        """Generate complete API reference page."""
        lines = [
            "# API Reference",
            "",
            "Complete API reference for Simpulse.",
            "",
            "## Table of Contents",
            ""
        ]
        
        # Group by module
        by_module = {}
        for doc in api_docs:
            module = doc.name.rsplit('.', 2)[0]
            if module not in by_module:
                by_module[module] = []
            by_module[module].append(doc)
        
        # Generate TOC
        for module in sorted(by_module.keys()):
            lines.append(f"- [{module}](#{module.replace('.', '')})")
        
        lines.append("")
        
        # Generate detailed docs
        for module in sorted(by_module.keys()):
            lines.append(f"## {module}")
            lines.append("")
            
            for doc in sorted(by_module[module], key=lambda x: x.name):
                lines.append(f"### `{doc.name}`")
                lines.append("")
                lines.append(f"**Type**: {doc.type}")
                lines.append("")
                lines.append("```python")
                lines.append(doc.signature)
                lines.append("```")
                lines.append("")
                
                if doc.docstring:
                    lines.append(doc.docstring)
                    lines.append("")
                
                if doc.parameters:
                    lines.append("**Parameters:**")
                    lines.append("")
                    for param in doc.parameters:
                        param_line = f"- `{param['name']}`"
                        if 'type' in param:
                            param_line += f" ({param['type']})"
                        if 'description' in param:
                            param_line += f": {param['description']}"
                        lines.append(param_line)
                    lines.append("")
                
                if doc.returns:
                    lines.append(f"**Returns:** {doc.returns}")
                    lines.append("")
                
                if doc.examples:
                    lines.append("**Examples:**")
                    lines.append("")
                    for example in doc.examples:
                        lines.append("```python")
                        lines.append(example)
                        lines.append("```")
                        lines.append("")
                
                lines.append(f"*Source: {doc.source_file}:{doc.line_number}*")
                lines.append("")
                lines.append("---")
                lines.append("")
        
        output_path.write_text('\n'.join(lines))
    
    def _generate_examples_page(self, output_path: Path):
        """Generate examples page."""
        lines = [
            "# Examples",
            "",
            "Practical examples of using Simpulse.",
            ""
        ]
        
        # Find example files
        if self.examples_dir.exists():
            for example_file in sorted(self.examples_dir.glob("*.py")):
                lines.append(f"## {example_file.stem.replace('_', ' ').title()}")
                lines.append("")
                
                # Read example content
                content = example_file.read_text()
                
                # Extract docstring
                try:
                    tree = ast.parse(content)
                    docstring = ast.get_docstring(tree)
                    if docstring:
                        lines.append(docstring)
                        lines.append("")
                except:
                    pass
                
                # Add code
                lines.append("```python")
                lines.append(content)
                lines.append("```")
                lines.append("")
        
        output_path.write_text('\n'.join(lines))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive documentation for Simpulse"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for documentation"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "html", "json"],
        default="markdown",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    if not args.output_dir:
        args.output_dir = args.project_root / "docs" / "generated"
    
    generator = DocumentationGenerator(args.project_root)
    
    if args.format == "markdown":
        success = generator.generate_markdown_docs(args.output_dir)
    else:
        logger.error(f"Format {args.format} not yet implemented")
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()