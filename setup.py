# setup.py
from setuptools import setup, find_packages

setup(
    name="task_offloading",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "networkx>=2.5.0",
        "pyyaml>=5.3.1",
        "matplotlib>=3.3.0",
        "tensorflow-macos>=2.4.0;platform_system=='Darwin' and platform_machine=='arm64'",
        "tensorflow>=2.4.0;platform_system!='Darwin' or platform_machine!='arm64'",
        "torch>=1.7.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.24.0",
        "jupyter>=1.0.0",
        "seaborn>=0.11.0",
        "plotly>=4.14.0",
        "tqdm>=4.50.0",
        "networkx>=2.5.0"
    ],
    python_requires=">=3.7",
    author="Your Name",
    author_email="your.email@example.com",
    description="Task offloading for vehicular edge computing",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)