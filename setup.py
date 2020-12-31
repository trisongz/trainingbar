import os, sys
import os.path

from setuptools import setup, find_packages

root = os.path.abspath(os.path.dirname(__file__))
package_name = "trainingbar"
packages = find_packages(
    include=[package_name, "{}.*".format(package_name)]
)

_locals = {}
with open(os.path.join(package_name, "_version.py")) as fp:
    exec(fp.read(), None, _locals)

version = _locals["__version__"]
binary_names = _locals["binary_names"]

with open(os.path.join(root, 'README.md'), 'rb') as readme:
    long_description = readme.read().decode('utf-8')

setup(
    name=package_name,
    version=version,
    description="trainingbar",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Tri Songz',
    author_email='ts@scontentenginex.com',
    keywords=['ml training', 'model training', 'tqdm', 'progress bar', 'monitoring', 'google cloud', 'tensorflow', 'pytorch'],
    url='http://github.com/trisongz/trainingbar',
    python_requires='>3.6',
    install_requires=[
        "rich",
        "tensorflow>=1.15.0",
        "psutil",
        "typer",
        "pysimdjson",
        "google-auth",
    ],
    packages=packages,
    package_data = {
        'json': ['*.json'],
    },
    include_package_data=True,
    extras_require={
        'tpu': ['tpunicorn', 'google-cloud-monitoring'],
        'gpu': ['gputil'],
        'test': ['pytest'],
    },
    entry_points={
        "console_scripts": [
            "tbar = trainingbar.cli:cli",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
    ],
)