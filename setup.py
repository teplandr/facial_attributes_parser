import io
import os
from setuptools import setup, find_packages


# Package meta-data.
description = "Segmentation of facial attributes."
url = "https://github.com/teplandr/facial_attributes_parser"
email = "teplandr@yandex.ru"
author = "Andrey Teplyakov"
requires_python = ">=3.0.0"
current_dir = os.path.abspath(os.path.dirname(__file__))


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


try:
    with open(os.path.join(current_dir, "requirements.txt"), encoding="utf-8") as f:
        required = f.read().split("\n")
except FileNotFoundError:
    required = []


setup(
    name="facial_attributes_parser",
    version="0.0.1",
    description=description,
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author=author,
    license="MIT",
    url=url,
    packages=find_packages(exclude=["tests", "docs", "images"]),
    install_requires=required,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
