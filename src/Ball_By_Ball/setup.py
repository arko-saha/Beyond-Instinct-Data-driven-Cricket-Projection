"""
Setup script for Cricket Run Prediction package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "documentation.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="cricket-run-prediction",
    version="1.0.0",
    author="Beyond Instinct Team",
    author_email="team@beyondinstinct.com",
    description="Advanced ML system for ball-by-ball cricket run prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/beyondinstinct/cricket-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="cricket, machine learning, prediction, sports analytics",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ],
        "mlflow": [
            "mlflow>=2.7.0",
        ],
        "shap": [
            "shap>=0.42.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cricket-predict=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.md"],
    },
    project_urls={
        "Bug Reports": "https://github.com/beyondinstinct/cricket-prediction/issues",
        "Source": "https://github.com/beyondinstinct/cricket-prediction",
        "Documentation": "https://github.com/beyondinstinct/cricket-prediction/blob/main/documentation.md",
    },
)