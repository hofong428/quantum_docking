from setuptools import setup, find_packages

setup(
    name="quantum_docking",
    version="0.1.0",
    description="Quantum computing-assisted molecular docking system",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "qiskit>=0.45.0",
        "pennylane>=0.32.0",
        "openfermion>=1.5.0",
        "rdkit-pypi>=2023.3.1",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
