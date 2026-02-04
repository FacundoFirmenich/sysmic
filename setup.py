from setuptools import setup, find_packages

setup(
    name="sfa",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "emcee>=3.0.0",
        "joblib>=1.0.0"
    ],
    author="Facundo Firmenich",
    author_email="f.firmenich@cedesur.org",
    description="Seismic Fractal Analysis Framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/FacundoFirmenich/SeismicFractalAnalysis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires='>=3.8',
)
