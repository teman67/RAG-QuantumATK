
from setuptools import setup, find_packages

setup(
    name="quantumatk-rag",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'flask>=2.3.2',
        'requests>=2.31.0',
        'beautifulsoup4>=4.12.2',
        'sentence-transformers>=2.2.2',
        'faiss-cpu>=1.7.4',
        'openai>=1.3.0',
        'numpy>=1.24.3',
        'tiktoken>=0.5.1',
        'lxml>=4.9.3',
    ],
    python_requires='>=3.8',
    description="RAG system for QuantumATK documentation and forum",
    author="Your Name",
    author_email="your.email@example.com",
)