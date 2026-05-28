from setuptools import setup, find_packages

setup(
    name="ArtC_UWB",
    version="1.0.0",
    description="Posicionamiento UWB Interior con Machine Learning y Optuna",
    author="Breimer",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "pandas>=3.0.0",
        "numpy>=2.4.0",
        "scipy>=1.17.0",
        "scikit-learn>=1.8.0",
        "matplotlib>=3.10.0",
        "torch>=2.12.0",
        "optuna>=3.5.0",
        "jupyter>=7.5.0",
    ],
)
