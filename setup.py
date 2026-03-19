"""
TinyTrain: Distributed LLM Training Framework
Setup script for pip installation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tinytrain",
    version="0.1.0",
    author="TinyTrain Contributors",
    description="A distributed LLM training framework from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DilpreetBansi/tinytrain",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "tinytrain-train=tinytrain.scripts.train_single_gpu:main",
        ],
    },
)
