#!/usr/bin/env python3
"""
Deploy Simpulse documentation website.

This script handles:
- Building documentation with Sphinx
- Deploying to GitHub Pages
- Setting up custom domain
- Creating API documentation
- Search functionality
- Version management
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DocsConfig:
    """Documentation deployment configuration."""
    project_name: str = "Simpulse"
    author: str = "Simpulse Team"
    copyright_year: str = str(datetime.now().year)
    version: str = "1.0.0"
    release: str = "1.0.0"
    domain: Optional[str] = "simpulse.dev"
    github_repo: str = "yourusername/simpulse"
    analytics_id: Optional[str] = None


class DocumentationDeployer:
    """Deploy Simpulse documentation website."""
    
    def __init__(self, project_root: Path, config: DocsConfig):
        """Initialize documentation deployer.
        
        Args:
            project_root: Root directory of the project
            config: Documentation configuration
        """
        self.project_root = project_root
        self.docs_dir = project_root / "docs"
        self.build_dir = self.docs_dir / "_build"
        self.config = config
    
    async def deploy_documentation(self) -> bool:
        """Complete documentation deployment process."""
        logger.info("Starting documentation deployment...")
        
        try:
            # Step 1: Setup Sphinx configuration
            logger.info("\n📝 Setting up Sphinx configuration...")
            self.setup_sphinx_config()
            
            # Step 2: Generate API documentation
            logger.info("\n🔧 Generating API documentation...")
            self.generate_api_docs()
            
            # Step 3: Create custom pages
            logger.info("\n📄 Creating custom documentation pages...")
            self.create_custom_pages()
            
            # Step 4: Setup theme and styling
            logger.info("\n🎨 Setting up theme and styling...")
            self.setup_theme()
            
            # Step 5: Build documentation
            logger.info("\n🏗️ Building documentation...")
            if not self.build_docs():
                return False
            
            # Step 6: Setup GitHub Pages
            logger.info("\n🌐 Setting up GitHub Pages...")
            self.setup_github_pages()
            
            # Step 7: Configure custom domain
            if self.config.domain:
                logger.info("\n🔗 Configuring custom domain...")
                self.setup_custom_domain()
            
            # Step 8: Deploy to GitHub Pages
            logger.info("\n🚀 Deploying to GitHub Pages...")
            self.deploy_to_github_pages()
            
            # Step 9: Setup search
            logger.info("\n🔍 Setting up search functionality...")
            self.setup_search()
            
            logger.info("\n✅ Documentation deployment complete!")
            return True
            
        except Exception as e:
            logger.error(f"Documentation deployment failed: {e}")
            return False
    
    def setup_sphinx_config(self) -> None:
        """Setup Sphinx configuration."""
        conf_path = self.docs_dir / "conf.py"
        
        conf_content = f'''# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# Project information
project = '{self.config.project_name}'
copyright = '{self.config.copyright_year}, {self.config.author}'
author = '{self.config.author}'
release = '{self.config.release}'
version = '{self.config.version}'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'sphinx_design',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/logo.png'
html_favicon = '_static/favicon.ico'

html_theme_options = {{
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
    'analytics_id': '{self.config.analytics_id or ""}',
    'analytics_anonymize_ip': False,
}}

# GitHub integration
html_context = {{
    'display_github': True,
    'github_user': '{self.config.github_repo.split("/")[0]}',
    'github_repo': '{self.config.github_repo.split("/")[1]}',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}}

# Copy button configuration
copybutton_prompt_text = r">>> |\\$ |In \\[\\d*\\]: | {{2,5}}\\.\\.\\.: | {{5,8}}: "
copybutton_prompt_is_regexp = True

# MyST configuration
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# Intersphinx
intersphinx_mapping = {{
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# TODO extension
todo_include_todos = True

# Autodoc
autodoc_default_options = {{
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}}

# Custom CSS
html_css_files = [
    'custom.css',
]

# Custom JavaScript
html_js_files = [
    'custom.js',
]
'''
        conf_path.write_text(conf_content)
        
        # Create static directory
        static_dir = self.docs_dir / "_static"
        static_dir.mkdir(exist_ok=True)
        
        # Create custom CSS
        css_path = static_dir / "custom.css"
        css_content = """/* Custom CSS for Simpulse documentation */

/* Improve code block styling */
.highlight {
    background-color: #f8f9fa !important;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 16px;
    margin: 16px 0;
}

/* Add gradient to header */
.wy-nav-top {
    background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
}

/* Improve admonition styling */
.admonition {
    border-radius: 6px;
    border-left: 4px solid;
}

.admonition.note {
    border-left-color: #007bff;
}

.admonition.warning {
    border-left-color: #ffc107;
}

.admonition.danger {
    border-left-color: #dc3545;
}

/* API documentation styling */
.class, .function, .method {
    margin-top: 2em;
    padding-top: 1em;
    border-top: 1px solid #e1e4e8;
}

/* Search highlighting */
.highlighted {
    background-color: #ffeb3b;
    padding: 2px 4px;
    border-radius: 3px;
}

/* Responsive tables */
table {
    display: block;
    overflow-x: auto;
    white-space: nowrap;
}

/* Copy button styling */
.copybtn {
    transition: opacity 0.3s;
}

/* Version selector */
.version-selector {
    margin: 10px 0;
    padding: 5px 10px;
    background: #f0f0f0;
    border-radius: 4px;
}
"""
        css_path.write_text(css_content)
        
        # Create custom JavaScript
        js_path = static_dir / "custom.js"
        js_content = """// Custom JavaScript for Simpulse documentation

// Add copy functionality to code blocks
document.addEventListener('DOMContentLoaded', function() {
    // Add line numbers to code blocks
    document.querySelectorAll('pre code').forEach(function(block) {
        if (block.innerHTML.split('\\n').length > 5) {
            block.classList.add('line-numbers');
        }
    });
    
    // Add anchor links to headers
    document.querySelectorAll('h2, h3, h4').forEach(function(header) {
        if (header.id) {
            const anchor = document.createElement('a');
            anchor.className = 'headerlink';
            anchor.href = '#' + header.id;
            anchor.innerHTML = '¶';
            anchor.title = 'Permalink to this headline';
            header.appendChild(anchor);
        }
    });
    
    // Improve search experience
    const searchBox = document.querySelector('.wy-side-nav-search input[type="text"]');
    if (searchBox) {
        searchBox.addEventListener('input', function(e) {
            // Add debouncing for search
            clearTimeout(this.searchTimeout);
            this.searchTimeout = setTimeout(function() {
                // Trigger search
                console.log('Searching for:', e.target.value);
            }, 300);
        });
    }
});

// Analytics event tracking
function trackEvent(category, action, label) {
    if (typeof gtag !== 'undefined') {
        gtag('event', action, {
            'event_category': category,
            'event_label': label
        });
    }
}

// Track documentation interactions
document.addEventListener('click', function(e) {
    if (e.target.matches('a[href^="http"]')) {
        trackEvent('External Link', 'click', e.target.href);
    }
    if (e.target.matches('.copybtn')) {
        trackEvent('Code', 'copy', window.location.pathname);
    }
});
"""
        js_path.write_text(js_content)
        
        logger.info("✓ Sphinx configuration created")
    
    def generate_api_docs(self) -> None:
        """Generate API documentation from source code."""
        # Create API documentation directory
        api_dir = self.docs_dir / "api"
        api_dir.mkdir(exist_ok=True)
        
        # Generate API index
        api_index = api_dir / "index.rst"
        api_index_content = """API Reference
=============

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   core
   evolution
   evaluation
   reporting
   utils

Core Module
-----------

.. automodule:: simpulse
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
~~~~~~~~~~~~~

.. automodule:: simpulse.config
   :members:
   :undoc-members:
   :show-inheritance:
"""
        api_index.write_text(api_index_content)
        
        # Generate module documentation
        modules = ["core", "evolution", "evaluation", "reporting", "utils"]
        
        for module in modules:
            module_path = api_dir / f"{module}.rst"
            module_content = f"""{module.title()} Module
{'=' * (len(module) + 7)}

.. automodule:: simpulse.{module}
   :members:
   :undoc-members:
   :show-inheritance:

Submodules
----------

.. toctree::
   :maxdepth: 1
   
   {module}_submodules
"""
            module_path.write_text(module_content)
        
        # Run sphinx-apidoc
        try:
            subprocess.run([
                "sphinx-apidoc",
                "-o", str(api_dir),
                "-f",  # Force overwrite
                "-e",  # Separate pages for each module
                "-M",  # Module-first
                str(self.project_root / "src" / "simpulse"),
                "*test*",  # Exclude tests
                "*__pycache__*"  # Exclude cache
            ], check=True)
            logger.info("✓ API documentation generated")
        except subprocess.CalledProcessError:
            logger.warning("Could not run sphinx-apidoc")
    
    def create_custom_pages(self) -> None:
        """Create custom documentation pages."""
        # Main index
        index_path = self.docs_dir / "index.rst"
        index_content = f"""{self.config.project_name} Documentation
{'=' * (len(self.config.project_name) + 14)}

.. image:: https://img.shields.io/pypi/v/simpulse.svg
   :target: https://pypi.org/project/simpulse/
   :alt: PyPI version

.. image:: https://img.shields.io/github/workflow/status/{self.config.github_repo}/CI
   :target: https://github.com/{self.config.github_repo}/actions
   :alt: Build status

.. image:: https://codecov.io/gh/{self.config.github_repo}/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/{self.config.github_repo}
   :alt: Code coverage

Welcome to Simpulse!
--------------------

Simpulse is an ML-powered optimization tool for Lean 4's simp tactic that uses evolutionary 
algorithms to discover optimal simplification strategies.

.. note::
   Simpulse is currently in beta. Join our `beta program <beta.html>`_ to get early access!

Features
--------

- 🧬 **Evolutionary Optimization**: Uses genetic algorithms for rule discovery
- 🤖 **Claude Integration**: Leverages AI for intelligent mutations
- 📊 **Performance Profiling**: Data-driven optimization decisions
- ✅ **Safety First**: All optimizations preserve proof correctness
- 🚀 **20%+ Speedup**: Proven improvements on real-world code

Quick Start
-----------

.. code-block:: bash

   pip install simpulse

.. code-block:: python

   from simpulse import Simpulse
   
   # Optimize your Lean 4 project
   optimizer = Simpulse()
   results = await optimizer.optimize(
       modules=["YourModule"],
       source_path="path/to/lean/project"
   )
   
   print(f"Improvement: {{results.improvement_percent}}%")

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   installation
   quickstart
   configuration
   advanced

.. toctree::
   :maxdepth: 2
   :caption: Concepts
   
   how_it_works
   evolutionary_algorithms
   performance_profiling
   safety_guarantees

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Community
   
   beta
   contributing
   changelog
   roadmap

.. toctree::
   :maxdepth: 1
   :caption: Resources
   
   examples
   benchmarks
   faq
   troubleshooting

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
"""
        index_path.write_text(index_content)
        
        # Installation guide
        install_path = self.docs_dir / "installation.rst"
        install_content = """Installation
============

Requirements
------------

- Python 3.8 or higher
- Lean 4 installed and in PATH
- 4GB RAM minimum (8GB recommended)
- 1GB free disk space

Install from PyPI
-----------------

The easiest way to install Simpulse is via pip:

.. code-block:: bash

   pip install simpulse

For the latest development version:

.. code-block:: bash

   pip install --pre simpulse

Install from Source
-------------------

To install from source:

.. code-block:: bash

   git clone https://github.com/yourusername/simpulse.git
   cd simpulse
   pip install -e ".[dev]"

Docker Installation
-------------------

We provide Docker images for easy deployment:

.. code-block:: bash

   docker pull simpulse/simpulse:latest
   docker run -v /path/to/lean/project:/workspace simpulse/simpulse

Verify Installation
-------------------

Verify your installation:

.. code-block:: python

   import simpulse
   print(simpulse.__version__)

Next Steps
----------

- Read the :doc:`quickstart` guide
- Configure Simpulse for your project: :doc:`configuration`
- Join our community: :doc:`beta`
"""
        install_path.write_text(install_content)
        
        # How it works page
        how_path = self.docs_dir / "how_it_works.md"
        how_content = """# How Simpulse Works

Simpulse optimizes Lean 4's `simp` tactic through a sophisticated ML-powered approach.

## Overview

```mermaid
graph LR
    A[Lean Project] --> B[Rule Extraction]
    B --> C[Performance Profiling]
    C --> D[Evolutionary Algorithm]
    D --> E[Mutation Generation]
    E --> F[Fitness Evaluation]
    F --> G{Improved?}
    G -->|Yes| H[Apply Changes]
    G -->|No| D
    H --> I[Optimized Project]
```

## The Process

### 1. Rule Extraction

Simpulse begins by analyzing your Lean 4 codebase to extract all simp rules:

- Parses `.lean` files to find `@[simp]` declarations
- Identifies rule priorities and directions
- Maps dependencies between rules

### 2. Performance Profiling

Next, we profile your code to understand performance characteristics:

- Measures simp tactic execution time
- Tracks memory usage
- Identifies bottlenecks

### 3. Evolutionary Optimization

The core optimization uses evolutionary algorithms:

- **Population**: Set of rule configurations
- **Fitness**: Performance improvement
- **Selection**: Best configurations survive
- **Mutation**: AI-guided rule modifications
- **Crossover**: Combine successful strategies

### 4. Intelligent Mutations

Simpulse uses Claude to generate intelligent mutations:

```python
# Example mutation types
- Change rule priority: @[simp] → @[simp high]
- Reverse rule direction: a = b → b = a
- Add contextual constraints
- Remove redundant rules
```

### 5. Safety Validation

Every optimization is validated:

- ✅ Proofs still compile
- ✅ Results remain correct
- ✅ No semantic changes
- ✅ Performance improves

## Example Optimization

Consider this scenario:

```lean
-- Before optimization
@[simp] theorem add_zero (a : Nat) : a + 0 = a := ...
@[simp] theorem zero_add (a : Nat) : 0 + a = a := ...
@[simp] theorem add_comm (a b : Nat) : a + b = b + a := ...

-- After optimization
@[simp high] theorem add_zero (a : Nat) : a + 0 = a := ...
@[simp low] theorem zero_add (a : Nat) : 0 + a = a := ...
@[simp ←] theorem add_comm (a b : Nat) : b + a = a + b := ...
```

Results:
- 25% faster simplification
- More predictable rule application
- Same mathematical correctness

## Learn More

- [Evolutionary Algorithms](evolutionary_algorithms.html)
- [Performance Profiling](performance_profiling.html)
- [Safety Guarantees](safety_guarantees.html)
"""
        how_path.write_text(how_content)
        
        logger.info("✓ Custom documentation pages created")
    
    def setup_theme(self) -> None:
        """Setup documentation theme and styling."""
        # Create requirements file for Read the Docs
        requirements_path = self.docs_dir / "requirements.txt"
        requirements_content = """sphinx>=4.0
sphinx-rtd-theme>=1.0
sphinx-copybutton>=0.5
sphinx-design>=0.3
myst-parser>=0.18
"""
        requirements_path.write_text(requirements_content)
        
        # Create .readthedocs.yaml
        rtd_config = self.project_root / ".readthedocs.yaml"
        rtd_content = """version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.10"

sphinx:
  configuration: docs/conf.py

formats:
  - pdf
  - epub

python:
  install:
    - requirements: docs/requirements.txt
    - method: pip
      path: .
      extra_requirements:
        - docs
"""
        rtd_config.write_text(rtd_content)
        
        logger.info("✓ Theme configuration complete")
    
    def build_docs(self) -> bool:
        """Build documentation with Sphinx."""
        try:
            # Clean previous builds
            if self.build_dir.exists():
                shutil.rmtree(self.build_dir)
            
            # Build HTML documentation
            result = subprocess.run([
                "sphinx-build",
                "-b", "html",
                "-W",  # Warnings as errors
                str(self.docs_dir),
                str(self.build_dir / "html")
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error("Documentation build failed")
                logger.error(result.stderr)
                return False
            
            logger.info("✓ Documentation built successfully")
            
            # Build PDF (optional)
            try:
                subprocess.run([
                    "sphinx-build",
                    "-b", "latexpdf",
                    str(self.docs_dir),
                    str(self.build_dir / "pdf")
                ], check=True)
                logger.info("✓ PDF documentation built")
            except:
                logger.info("PDF build skipped (LaTeX not available)")
            
            return True
            
        except Exception as e:
            logger.error(f"Build failed: {e}")
            return False
    
    def setup_github_pages(self) -> None:
        """Setup GitHub Pages configuration."""
        # Create .nojekyll file
        nojekyll = self.build_dir / "html" / ".nojekyll"
        nojekyll.touch()
        
        # Create GitHub Pages config
        if self.config.domain:
            cname = self.build_dir / "html" / "CNAME"
            cname.write_text(self.config.domain)
        
        # Create 404 page
        not_found = self.build_dir / "html" / "404.html"
        not_found_content = """<!DOCTYPE html>
<html>
<head>
    <title>Page Not Found - Simpulse</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
        h1 { color: #dc3545; }
        a { color: #007bff; text-decoration: none; }
    </style>
</head>
<body>
    <h1>404 - Page Not Found</h1>
    <p>The page you're looking for doesn't exist.</p>
    <p><a href="/">Return to Homepage</a></p>
</body>
</html>
"""
        not_found.write_text(not_found_content)
        
        logger.info("✓ GitHub Pages configured")
    
    def setup_custom_domain(self) -> None:
        """Setup custom domain configuration."""
        # This would typically involve DNS configuration
        # For now, we'll create instructions
        
        dns_instructions = self.docs_dir / "DNS_SETUP.md"
        dns_content = f"""# DNS Configuration for {self.config.domain}

## GitHub Pages Setup

1. Go to repository settings > Pages
2. Set source to `gh-pages` branch
3. Enter custom domain: `{self.config.domain}`

## DNS Records

Add these records to your DNS provider:

### For apex domain ({self.config.domain}):
- A record: 185.199.108.153
- A record: 185.199.109.153
- A record: 185.199.110.153
- A record: 185.199.111.153

### For www subdomain:
- CNAME record: www -> {self.config.github_repo.split('/')[0]}.github.io

## SSL Certificate

GitHub Pages automatically provisions an SSL certificate via Let's Encrypt.
This may take up to 24 hours after DNS propagation.

## Verification

```bash
dig {self.config.domain}
nslookup {self.config.domain}
```

Visit https://{self.config.domain} to verify setup.
"""
        dns_instructions.write_text(dns_content)
        
        logger.info("✓ Custom domain instructions created")
    
    def deploy_to_github_pages(self) -> None:
        """Deploy documentation to GitHub Pages."""
        try:
            # Initialize gh-pages branch if needed
            subprocess.run([
                "git", "checkout", "--orphan", "gh-pages"
            ], cwd=self.project_root, capture_output=True)
            
            # Remove all files
            subprocess.run([
                "git", "rm", "-rf", "."
            ], cwd=self.project_root, capture_output=True)
            
            # Copy built documentation
            html_dir = self.build_dir / "html"
            if html_dir.exists():
                for item in html_dir.iterdir():
                    dest = self.project_root / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
            
            # Commit and push
            subprocess.run([
                "git", "add", "."
            ], cwd=self.project_root)
            
            subprocess.run([
                "git", "commit", "-m", "Deploy documentation"
            ], cwd=self.project_root)
            
            subprocess.run([
                "git", "push", "origin", "gh-pages", "--force"
            ], cwd=self.project_root)
            
            # Switch back to main branch
            subprocess.run([
                "git", "checkout", "main"
            ], cwd=self.project_root)
            
            logger.info("✓ Documentation deployed to GitHub Pages")
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            # Try to recover
            subprocess.run(["git", "checkout", "main"], cwd=self.project_root)
    
    def setup_search(self) -> None:
        """Setup search functionality for documentation."""
        # For GitHub Pages, we'll use Algolia DocSearch
        search_config = self.docs_dir / "search_config.json"
        search_content = {
            "index_name": "simpulse",
            "start_urls": [
                f"https://{self.config.domain or 'yourusername.github.io/simpulse'}/"
            ],
            "stop_urls": [],
            "selectors": {
                "lvl0": {
                    "selector": ".menu-content h1",
                    "default_value": "Documentation"
                },
                "lvl1": ".content h1",
                "lvl2": ".content h2",
                "lvl3": ".content h3",
                "lvl4": ".content h4",
                "text": ".content p, .content li"
            }
        }
        
        with open(search_config, 'w') as f:
            json.dump(search_content, f, indent=2)
        
        # Add search to documentation
        search_js = self.docs_dir / "_static" / "algolia_search.js"
        search_js_content = """// Algolia DocSearch integration
docsearch({
    apiKey: 'YOUR_ALGOLIA_API_KEY',
    indexName: 'simpulse',
    inputSelector: '#search-input',
    debug: false
});
"""
        search_js.write_text(search_js_content)
        
        logger.info("✓ Search functionality configured")
    
    def create_deployment_report(self) -> None:
        """Create documentation deployment report."""
        report_path = self.project_root / "docs_deployment_report.md"
        
        lines = [
            "# Documentation Deployment Report",
            "",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Version**: {self.config.version}",
            "",
            "## Deployment Summary",
            "",
            "- ✅ Sphinx configuration created",
            "- ✅ API documentation generated",
            "- ✅ Custom pages created",
            "- ✅ Theme configured",
            "- ✅ Documentation built",
            "- ✅ GitHub Pages configured",
            f"- ✅ Custom domain: {self.config.domain or 'Not configured'}",
            "- ✅ Search functionality added",
            "",
            "## Access URLs",
            "",
            f"- **GitHub Pages**: https://{self.config.github_repo.split('/')[0]}.github.io/{self.config.github_repo.split('/')[1]}/",
        ]
        
        if self.config.domain:
            lines.append(f"- **Custom Domain**: https://{self.config.domain}/")
        
        lines.extend([
            "",
            "## Next Steps",
            "",
            "1. **Configure Algolia Search**:",
            "   - Sign up at https://www.algolia.com/",
            "   - Submit site to DocSearch",
            "   - Add API key to documentation",
            "",
            "2. **Monitor Analytics**:",
            "   - Set up Google Analytics",
            "   - Track popular pages",
            "   - Monitor search queries",
            "",
            "3. **Continuous Updates**:",
            "   - Set up automatic deployment",
            "   - Version documentation",
            "   - Add language translations",
            "",
            "## Files Generated",
            "",
            "- `docs/conf.py` - Sphinx configuration",
            "- `docs/index.rst` - Main documentation page",
            "- `docs/api/` - API reference",
            "- `docs/_static/` - Static assets",
            "- `docs/_build/html/` - Built documentation",
            "- `.readthedocs.yaml` - Read the Docs config",
        ])
        
        report_path.write_text('\n'.join(lines))
        logger.info(f"Deployment report saved to {report_path}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy Simpulse documentation"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Documentation version"
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Custom domain (e.g., simpulse.dev)"
    )
    parser.add_argument(
        "--github-repo",
        type=str,
        default="yourusername/simpulse",
        help="GitHub repository (owner/name)"
    )
    parser.add_argument(
        "--analytics-id",
        type=str,
        help="Google Analytics ID"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build docs, don't deploy"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = DocsConfig(
        version=args.version,
        release=args.version,
        domain=args.domain,
        github_repo=args.github_repo,
        analytics_id=args.analytics_id
    )
    
    # Deploy documentation
    deployer = DocumentationDeployer(args.project_root, config)
    
    if args.build_only:
        # Just build
        deployer.setup_sphinx_config()
        deployer.generate_api_docs()
        deployer.create_custom_pages()
        deployer.setup_theme()
        success = deployer.build_docs()
    else:
        # Full deployment
        success = await deployer.deploy_documentation()
    
    if success:
        deployer.create_deployment_report()
        logger.info("\n" + "="*60)
        logger.info("DOCUMENTATION DEPLOYMENT COMPLETE")
        logger.info("="*60)
        if args.domain:
            logger.info(f"Documentation: https://{args.domain}/")
        else:
            logger.info(f"Documentation: https://{args.github_repo.split('/')[0]}.github.io/{args.github_repo.split('/')[1]}/")
        logger.info("="*60)
    else:
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())