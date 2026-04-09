"""Build configuration for Cython extensions.

Run ``python setup.py build_ext --inplace`` to compile the ``_core``
extension in-place for development.  For production builds the
pyproject.toml build-system declaration ensures Cython is available.
"""

import platform
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# --- OpenMP flags (platform-dependent) ---
if sys.platform == "darwin":
    # macOS: clang needs -Xpreprocessor and libomp from Homebrew
    omp_compile = ["-Xpreprocessor", "-fopenmp"]
    omp_link = ["-lomp"]
elif platform.system() == "Windows":
    omp_compile = ["/openmp"]
    omp_link = []
else:
    # Linux / GCC
    omp_compile = ["-fopenmp"]
    omp_link = ["-fopenmp"]

extensions = [
    Extension(
        "phone_similarity._core",
        sources=["src/phone_similarity/_core.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=omp_compile,
        extra_link_args=omp_link,
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "language_level": "3",
        },
    ),
)
