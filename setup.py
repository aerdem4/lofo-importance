from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='lofo-importance',
    version='0.3.0',
    url="https://github.com/aerdem4/lofo-importance",
    author="Ahmet Erdem",
    author_email="ahmeterd4@gmail.com",
    description="Leave One Feature Out Importance",
    keywords="feature importance selection explainable data-science machine-learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
