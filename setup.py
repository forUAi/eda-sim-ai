"""
Setup script for EDA-SIM-AI: Learning-Based Chip Placement Engine
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f 
                       if line.strip() and not line.startswith("#")]

setup(
    name="eda-sim-ai",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A state-of-the-art chip placement system using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/eda-sim-ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.2.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=6.2.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinx-autodoc-typehints>=1.23.0",
        ],
        "openroad": [
            # OpenROAD integration requires separate installation
            # See scripts/install_openroad.sh
        ],
    },
    entry_points={
        "console_scripts": [
            "eda-sim-ai=src.cli:main",
            "eda-sim-train=src.training.train:main",
            "eda-sim-benchmark=src.evaluation.benchmark:main",
            "eda-sim-place=src.inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "eda_sim_ai": [
            "configs/*.yaml",
            "data/examples/*",
        ],
    },
    zip_safe=False,
)