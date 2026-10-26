from setuptools import setup, find_packages

setup(
    name="hydeph",
    version="0.6.0",
    description="HyDePh: High-Definition Hybrid Defect Photoluminescence & Anharmonic Spectroscopy",
    author="Quantum Materials & Machine Learning Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22.0,<3",
        "scipy>=1.8.0,<2",
        "matplotlib>=3.5.0,<4",
        "ase>=3.22.0,<4",
        "calorine>=2.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "ruff>=0.4",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hydeph=hydeph.pipeline:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
