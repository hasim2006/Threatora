from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="threatora",
    version="1.0.0",
    author="Mohammad Hasim",
    author_email="juug25btech29109@jainuniversity.ac.in",
    description="Advanced Cyber Attack & Intrusion Detection System Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hasim2006/Threatora",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "threatora=main_unified_pipeline:main",
        ],
    },
)
