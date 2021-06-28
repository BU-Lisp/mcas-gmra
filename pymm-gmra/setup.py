# SYSTEM IMPORTS
from typing import List, Dict
from torch.utils import cpp_extension
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
import os
import platform
import re
import subprocess
import sys
import sysconfig


_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS

"""
    Most code in this file is shamelessly adapted from https://github.com/pybind/cmake_example/blob/master/setup.py
"""

class CMakeExtension(Extension):
    def __init__(self, name: str, src_dir: str = None):
        super().__init__(name, sources=list())
        self.src_dir = src_dir
        if self.src_dir is None:
            self.src_dir = ""
        self.src_dir = os.path.abspath(self.src_dir)


class CMakeBuild(build_ext):

    def run(self) -> None:
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("ERROR: cmake not found on this system but is required")

        if platform.system() == "Windows":
            cmake_version: str = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < "3.0.0":
                raise RuntimeError("ERROR: cmake > 3.0.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext) -> None:
        ext_dir: str = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not ext_dir.endswith(os.path.sep):
            ext_dir += os.path.sep

        torch_path: str = os.path.join(cpp_extension.include_paths()[0], "..")

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + ext_dir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      "-DCMAKE_PREFIX_PATH='%s'" % torch_path,
                      "-DCMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES=%s" % sysconfig.get_paths()['include']]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), ext_dir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        env["CC"] = subprocess.check_output("which gcc", stderr=subprocess.STDOUT,
                                            shell=True).strip().rstrip()
        env["CXX"] = subprocess.check_output("which g++", stderr=subprocess.STDOUT,
                                             shell=True).strip().rstrip()

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.src_dir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp, env=env)


setup(
    name = "mcas-gmra",
    version="0.0.1",
    author="Andrew Wood",
    author_email="aewood@bu.edu",
    description="Adaptation of GMRA (https://mauromaggioni.duckdns.org/code/) to Python in C++ using MCAS, pybind11, CMake, and libtorch",
    long_description="",
    ext_modules=[CMakeExtension("mcas-gmra")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)

