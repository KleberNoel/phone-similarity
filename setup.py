"""Build configuration for Cython extensions.

Run ``python setup.py build_ext --inplace`` to compile the ``_core``
extension in-place for development.  For production builds the
pyproject.toml build-system declaration ensures Cython is available.
"""

import os
import platform
import sys
import tempfile

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


def _has_openmp() -> bool:
    """Try to compile a small C file with OpenMP to detect availability."""
    from distutils.ccompiler import new_compiler
    from distutils.errors import CompileError, LinkError

    compiler = new_compiler()
    if "CC" in os.environ:
        compiler.set_executables(compiler=os.environ["CC"])

    src = b"#include <omp.h>\nint main(void) { return omp_get_max_threads(); }\n"
    tmpdir = tempfile.mkdtemp()
    src_path = os.path.join(tmpdir, "test_omp.c")
    with open(src_path, "wb") as f:
        f.write(src)

    if sys.platform == "darwin":
        compile_flags = ["-Xpreprocessor", "-fopenmp"]
        link_flags = ["-lomp"]
    elif platform.system() == "Windows":
        compile_flags = ["/openmp"]
        link_flags = []
    else:
        compile_flags = ["-fopenmp"]
        link_flags = ["-fopenmp"]

    # Inject CFLAGS / LDFLAGS from environment so Homebrew libomp is found
    env_cflags = os.environ.get("CFLAGS", "").split()
    env_ldflags = os.environ.get("LDFLAGS", "").split()

    try:
        objs = compiler.compile(
            [src_path],
            output_dir=tmpdir,
            extra_preargs=env_cflags + compile_flags,
        )
        compiler.link_executable(
            objs,
            os.path.join(tmpdir, "test_omp"),
            extra_preargs=env_ldflags + link_flags,
        )
        return True
    except (CompileError, LinkError, OSError):
        return False


# --- Detect OpenMP ---
have_openmp = _has_openmp()

if have_openmp:
    if sys.platform == "darwin":
        omp_compile = ["-Xpreprocessor", "-fopenmp"]
        omp_link = ["-lomp"]
    elif platform.system() == "Windows":
        omp_compile = ["/openmp"]
        omp_link = []
    else:
        omp_compile = ["-fopenmp"]
        omp_link = ["-fopenmp"]
else:
    omp_compile = []
    omp_link = []

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
