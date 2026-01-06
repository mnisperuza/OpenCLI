"""
Bert CLI — Setup
Installable via: pip install bert-cli
Cross-platform: Windows, Linux, macOS"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')

# Core dependencies (all platforms)
install_requires = [
    # ML Core
    "torch>=2.1.0",
    "transformers>=4.42.0",
    "accelerate>=0.27.0",
    
    # Model Hub
    "huggingface-hub>=0.20.0",
    "hf_xet>=1.1.0",
    "safetensors>=0.4.0",
    
    # Tokenizers (required by models)
    "sentencepiece>=0.1.99",
    "einops>=0.7.0",
    
    # Utilities
    "numpy>=1.24.0,<2.0.0",
]

setup(
    name="bert-cli",
    version="1.0.0",
    description="Bert — A CLI Framework by Matias Nisperuza",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Matias Nisperuza",
    author_email="mnisperuza1102@gmail.com",
    url="https://github.com/mnisperuza/bert-cli",
    project_urls={
        "Homepage": "https://mnisperuza.github.io/bert-cli/",
        "Bug Tracker": "https://github.com/mnisperuza/bert-cli/issues",
        "Documentation": "https://github.com/mnisperuza/bert-cli#readme",
        "Source": "https://github.com/mnisperuza/bert-cli",
    },
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        # Linux quantization support
        "linux": [
            "bitsandbytes>=0.43.0",
        ],
        # Performance optimizations
        "perf": [
            "xformers>=0.0.25",
        ],
        # Full install
        "full": [
            "bitsandbytes>=0.43.0",
            "xformers>=0.0.25",
        ],
    },
    entry_points={
        "console_scripts": [
            "bert=bert.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
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
    keywords="ai assistant llm local cli qwen amphydia",
)
