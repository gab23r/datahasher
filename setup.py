import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="datahasher",
    version="0.1.0",
    author="Gabriel Robin",
    description="Toolkit to develop data pipelines in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gab23r/datahasher",
    packages=setuptools.find_packages(),
    install_requires=["sqlmodel", "polars"],
    extras_require={
        "front": ["ipyvuetable", "ipyevents"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)