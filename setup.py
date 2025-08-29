#!/usr/bin/env python3
"""
Setup script for ChainBreak
Blockchain Forensic Analysis Tool
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip()
                    and not line.startswith("#")]

setup(
    name="chainbreak",
    version="1.0.0",
    author="ChainBreak Team",
    author_email="support@chainbreak.com",
    description="Blockchain Forensic Analysis Tool for detecting illicit cryptocurrency activity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ChainBreak",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Law Enforcement",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chainbreak=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords=[
        "blockchain",
        "forensics",
        "cryptocurrency",
        "bitcoin",
        "analysis",
        "anomaly-detection",
        "risk-scoring",
        "neo4j",
        "graph-database",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ChainBreak/issues",
        "Source": "https://github.com/yourusername/ChainBreak",
        "Documentation": "https://github.com/yourusername/ChainBreak/wiki",
    },
)
