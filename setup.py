from setuptools import setup, find_packages

setup(
    name="voltinfer",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "psutil>=5.9.0"
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
        "llamacpp": ["llama-cpp-python>=0.2.70"],
        "transformers": ["torch>=2.0.0", "transformers>=4.36.0", "accelerate>=0.25.0"]
    },
    entry_points={
        "console_scripts": [
            "voltinfer=engine.cli:main",
        ],
    },
)
