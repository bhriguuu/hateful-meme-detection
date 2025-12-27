"""
Setup script for Hateful Meme Detection package.

Install in development mode:
    pip install -e .

Install with all dependencies:
    pip install -e .[all]
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hateful-meme-detection",
    version="1.0.0",
    author="Bhrigu Anilkumar, Deepa Chandrasekar, Arshpreet Kaur",
    author_email="bhriguanilkumar@gmail.com",
    description="Multimodal Hateful Meme Detection using CLIP with Cross-Attention Fusion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hateful-meme-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "Pillow>=9.0.0",
        "numpy>=1.23.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "train": [
            "albumentations>=1.3.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "pandas>=1.5.0",
        ],
        "discord": [
            "discord.py>=2.0.0",
            "requests>=2.28.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "all": [
            "albumentations>=1.3.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "pandas>=1.5.0",
            "discord.py>=2.0.0",
            "requests>=2.28.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-hateful-meme=src.train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
