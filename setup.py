from setuptools import setup, find_packages

VERSION = "0.1.0"
DESCRIPTION = (
    "A lightweight deep learning framework for general use cases based on PyTorch"
)
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="pytorch-learn",
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "tqdm",
        "pytorch-nlp",
        "numpy>=1.19.2",
        "scipy>=1.2.1",
        "matplotlib>=3.0.3",
    ],
    author="carefree0910",
    author_email="syameimaru.saki@gmail.com",
    url="https://github.com/pku-aiic/pytorch-learn",
    download_url=f"https://github.com/pku-aiic/pytorch-learn/archive/v{VERSION}.tar.gz",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords="python computer-vision natural-language-processing machine-learning PyTorch",
)
