import os

import setuptools


def generate_all_paths(base_path):
    """Recursively generate glob patterns for all nested directories."""
    paths = []

    for root, dirs, _ in os.walk(base_path):
        if dirs:
            for d in dirs:
                dir_path = os.path.join(root, d).replace(base_path, "").lstrip("/")
                paths.append(f"{dir_path}/*")

    return paths


def package_files(directory):
    """Retrieve all nested paths and files from the specified directory."""
    paths = []
    for path, _, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths


framework_models_paths = generate_all_paths("framework_models")
stub_files = generate_all_paths("typestub-database")


with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()

setuptools.setup(
    name="headergen",
    version="2.0.0",
    description="HeaderGen: Automated cell header generator",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ashwinprasadme/headergen",
    author="Ashwin Prasad SV",
    author_email="ashwinprasad.me@gmail.com",
    packages=setuptools.find_packages(),
    package_data={
        "framework_models": framework_models_paths,  # Include any JSON files in the package directory
        "typestub-database": stub_files,
    },
    install_requires=[requirements],
    entry_points={
        "console_scripts": [
            "headergen = headergen.cli:cli",
        ],
    },
)
