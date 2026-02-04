from setuptools import setup, find_packages

setup(
    name="sysmic",
    version="6.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "emcee>=3.0.0",
        "joblib>=1.0.0",
        "dynesty>=1.0.0",
        "corner>=2.2.0"
    ],
    author="Facundo Firmenich",
    author_email="f.firmenich@cedesur.org",
    description="Sysmic: An Open-Source Intelligence Framework for High-Precision Fractal Tomography",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/facundofirmenich/sysmic",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires='>=3.8',
)
