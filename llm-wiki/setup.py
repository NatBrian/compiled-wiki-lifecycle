from setuptools import setup, find_packages

setup(
    name="llm-wiki",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6",
        "requests>=2",
    ],
    extras_require={
        "anthropic": ["anthropic>=0.30"],
    },
    entry_points={
        "console_scripts": [
            "llm-wiki=llm_wiki.cli:main",
        ],
    },
    python_requires=">=3.10",
)
