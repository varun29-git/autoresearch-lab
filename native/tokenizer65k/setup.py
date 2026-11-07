from pathlib import Path
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

setup(
    name="terminus-tokenizer-native",
    version="0.1.0",
    ext_modules=[Pybind11Extension(
        "_terminus_tokenizer_native", [str(Path(__file__).parent / "encoder.cpp")],
        cxx_std=17, extra_compile_args=["-O3"],
    )],
    cmdclass={"build_ext": build_ext},
)
