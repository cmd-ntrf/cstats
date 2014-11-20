import numpy as np

from setuptools import setup
from setuptools import setup, Extension
from Cython.Distutils import build_ext

kendalltau = Extension('cstats.kendalltau',
                       sources = ['cstats/kendalltau.pyx'],
                       include_dirs = [np.get_include()],)

gibbs = Extension('cstats.gibbs',
                  sources = ['cstats/gibbs.pyx'],
                 include_dirs = [np.get_include()],)
setup(
        packages = ['cstats'],
        name = "cstats",
        version = "0.1",
        ext_modules = [kendalltau, gibbs],
        cmdclass = {'build_ext': build_ext},
        install_requires=[
            "numpy",
            "cython"
        ]
)

