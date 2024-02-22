import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dema",
    version="0.2.2",
    author="Gabriel Robin",
    description="Data Engine MAnagment tool for python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gab23r/dema",
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
    package_data={"": ["front/style.css", "concepts_desc.csv"]},
    python_requires=">=3.11",
)
