import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "requirements.txt")) as f:
    install_requires = f.readlines()

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    readme = f.read()

with open(os.path.join(here, "uips", "version.py"), encoding="utf-8") as f:
    version = f.read()
version = version.split("=")[-1].strip().strip('"').strip("'")


setup(
    name="uips",
    version=version,
    description="Reduce a large and high-dimensional dataset by downselecting data uniformly in phase space",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/NREL/Phase-space-sampling",
    author="Malik Hassanaly",
    license="BSD 3-Clause",
    package_dir={"uips": "uips"},
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    package_data={"": ["input2D", "input2D_bins"]},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=install_requires,
)
