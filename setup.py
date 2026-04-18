from setuptools import setup, find_packages

setup(
    name="flowneg",
    version="0.1.0",
    description=(
        "FlowNeg: Exposure-Grounded Negative Generation with Controllable "
        "Hardness via Conditional Flow Matching"
    ),
    packages=find_packages(include=["flowneg", "flowneg.*"]),
    python_requires=">=3.10",
    install_requires=[
        "recbole>=1.2.0",
        "torch>=2.0.0",
        "numpy>=1.24.0,<2.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "faiss-cpu>=1.7.0",
        "tqdm>=4.60.0",
        "tensorboard>=2.10.0",
        "pyyaml>=6.0",
    ],
)
