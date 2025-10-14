"""Setup script for VEAH LLM"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="veah-llm",
    version="1.0.0",
    author="VEAH AI",
    description="Solana-Native Language Model for Blockchain Intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/veah-ai/veah-llm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "solana>=0.30.0",
        "pydantic>=2.5.0",
    ],
    entry_points={
        "console_scripts": [
            "veah=app:main",
            "veah-train=model.training:main",
            "veah-api=api.app:main",
        ],
    },
)