from setuptools import setup, find_packages

setup(
    name="neural-nets",
    version="0.0.0",
    description="Implementation of Various Neural Network Architectures",
    author="Anton1o-I",
    author_email="a.iniguez21@gmail.com",
    packages=find_packages(),
    license="LICENSE.txt",
    long_description=open("README.md").read(),
    install_requires=["numpy", "tensorflow", "pandas", "keras"],
    tests_require=["pytest"],
)
