from setuptools import setup, find_packages

setup(
    name="PathCondRAG",
    version="0.1.0",
    description="PathCondRAG: path-conditioned query decomposition for multi-hop RAG (PC3 + optional MPCE)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
