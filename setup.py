from setuptools import setup, find_packages

setup(
    name="training-quality-efficiency",
    version="1.0.0",
    description="Training Quality Determines Efficiency Boundaries in Test-Time Reasoning",
    author="Anonymous",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
)
