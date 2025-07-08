#!/usr/bin/env python
"""Setup script for Simpulse - for compatibility with older pip versions."""

from pathlib import Path

from setuptools import find_packages, setup

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="simpulse",
    version="1.0.0",
    description="Lightning-fast optimizer for Lean 4 simp rules - reduce proof search time by 2.83x",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Bright Liu",
    author_email="brightliu@college.harvard.edu",
    url="https://github.com/Bright-L01/simpulse",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "click>=8.1.7",
        "rich>=13.7.0",
    ],
    extras_require={
        "memory": ["psutil>=5.9.0"],  # Optional memory monitoring
    },
    entry_points={
        "console_scripts": [
            "simpulse=simpulse.cli:main",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Compilers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
    ],
    keywords="lean4 theorem-proving optimization simp performance",
    license="MIT",
)
