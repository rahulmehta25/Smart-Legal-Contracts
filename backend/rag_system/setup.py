"""
Setup configuration for the RAG Arbitration Detection System.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements from requirements.txt
requirements_path = this_directory / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().splitlines()
    # Filter out comments and empty lines
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="rag-arbitration-detection",
    version="1.0.0",
    description="A comprehensive RAG-based system for detecting arbitration clauses in legal documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="RAG Arbitration Team",
    author_email="contact@ragarbitration.com",
    url="https://github.com/arbitration-detection/rag-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Legal Industry",
        "Intended Audience :: Developers",
        "Topic :: Text Processing :: Legal Document Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9,<4.0",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.11.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
        ],
        "blockchain": [
            "web3>=6.11.3",
            "py-solc-x>=1.12.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "performance": [
            "nvidia-ml-py>=11.525.0",
            "psutil>=5.9.6",
            "memory-profiler>=0.61.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "rag-arbitration=src.api.main:main",
            "rag-train=src.core.training:main",
            "rag-evaluate=src.core.evaluation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "rag_system": [
            "config/*.yaml",
            "config/*.json", 
            "data/knowledge_base/*.json",
            "models/embeddings/*.bin",
            "models/classifiers/*.pkl",
        ],
    },
    zip_safe=False,
    keywords=[
        "legal-tech",
        "arbitration",
        "nlp",
        "rag",
        "document-analysis",
        "machine-learning",
        "ai",
        "legal-documents",
        "contract-analysis"
    ],
    project_urls={
        "Bug Reports": "https://github.com/arbitration-detection/rag-system/issues",
        "Source": "https://github.com/arbitration-detection/rag-system",
        "Documentation": "https://rag-arbitration-docs.readthedocs.io/",
    },
)