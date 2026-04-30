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


def _native_compile_args() -> list[str]:
    """Return platform-appropriate flags for maximum optimisation.

    ``-O3`` enables all standard optimisations (loop unrolling, auto-
    vectorisation, inlining).  ``-march=native`` emits SIMD instructions
    (AVX2/AVX-512 on modern x86, NEON on ARM) tailored to the build host.
    Both are skipped on Windows (MSVC uses /O2 instead).

    Note: wheels built with ``-march=native`` are not portable across
    microarchitectures.  For distributable wheels, remove ``-march=native``
    and rely on ``-O3`` alone (the CI build matrix sets CFLAGS explicitly).
    """
    is_ci = os.environ.get("CIBUILDWHEEL") or os.environ.get("CI")
    if platform.system() == "Windows":
        # MSVC: /O2 = full optimisation; /arch:AVX2 enables 256-bit SIMD
        # (equivalent of -march=native on GCC/Clang for modern x86).
        # Omit /arch:AVX2 on CI to keep distributable wheels portable.
        flags = ["/O2"]
        if not is_ci:
            flags.append("/arch:AVX2")
        return flags
    flags = ["-O3"]
    # -march=native is only meaningful for local dev / benchmarking builds.
    # Skip on CI (detected via environment) to keep wheels portable.
    if not is_ci:
        flags.append("-march=native")
    return flags


def _cpp_compile_args() -> list[str]:
    is_ci = os.environ.get("CIBUILDWHEEL") or os.environ.get("CI")
    if platform.system() == "Windows":
        flags = ["/O2", "/std:c++17"]
        if not is_ci:
            flags.append("/arch:AVX2")
        return flags
    base = ["-O3", "-std=c++17"]
    if not is_ci:
        base.append("-march=native")
    return base


native_compile = _native_compile_args()
cpp_compile = _cpp_compile_args()

extensions = [
    Extension(
        "phone_similarity._core",
        sources=["src/phone_similarity/_core.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=native_compile + omp_compile,
        extra_link_args=omp_link,
    ),
    Extension(
        "phone_similarity._beam_cpp",
        sources=["src/phone_similarity/_beam_cpp.cpp"],
        language="c++",
        extra_compile_args=cpp_compile,
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
