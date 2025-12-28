"""
Bert CLI — Setup
Installable via: pip install bert-cli
Cross-platform: Windows, Linux, macOS
"""

from setuptools import setup, find_packages
from pathlib import Path
import platform

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')

# Base dependencies (all platforms)
install_requires = [
    "torch>=2.0.0",
    "transformers>=4.40.0",
    "accelerate>=0.27.0",
    "huggingface-hub>=0.20.0",
    "hf_xet>=1.1.0",  # Fast model downloads
    "safetensors>=0.4.0",
    "numpy>=1.24.0,<2.0.0",
]

setup(
    name="bert-cli",
    version="1.0.0b",
    description="Bert — A calm, local AI assistant by Biwa Industries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Biwa",
    author_email="biwaindustries@gmail.com",
    url="https://github.com/mnisperuza/bert-cli",
    project_urls={
        "Bug Tracker": "https://github.com/mnisperuza/bert-cli/issues",
        "Documentation": "https://github.com/mnisperuza/bert-cli#readme",
        "Source": "https://github.com/mnisperuza/bert-cli",
    },
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "linux": [
            "bitsandbytes>=0.43.0",  # Full quantization support
        ],
        "web": [
            "duckduckgo-search>=6.0.0",
            "requests>=2.31.0",
            "beautifulsoup4>=4.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bert=bert.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="ai assistant llm local cli qwen bert",
)
