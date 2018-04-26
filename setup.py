from setuptools import setup, find_packages

version = "0.1.0"

with open("./README.md") as fd:
    long_description = fd.read()

setup(
    name="pmetsurf",
    version=version,
    description=
    "ParaMETric SURFaces - a library for manipulating 3D parametric surfaces",
    long_description=long_description,
    install_requires=[
        "numpy",
        "scipy"
    ],
    author="Kwanghun Chung Lab",
    packages=["pmetsurf"],
    url="https://github.com/chunglabmit/pmetsurf",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Programming Language :: Python :: 3.5',
    ]
)