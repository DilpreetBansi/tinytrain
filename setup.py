from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tinytrain",
    version="0.2.0",
    author="Dilpreet Singh Bansi",
    description="Data, tensor and pipeline parallelism for GPT training, built on torch.distributed",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DilpreetBansi/tinytrain",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=["torch>=2.1", "numpy>=1.24"],
    extras_require={"dev": ["pytest>=7.4", "matplotlib>=3.7"]},
    classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License"],
)
