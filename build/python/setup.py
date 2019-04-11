##
##  HMLP (High-Performance Machine Learning Primitives)
##
##  Copyright (C) 2014-2017, The University of Texas at Austin
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program. If not, see the LICENSE file.
##

#from distutils.core import setup
#from distutils.extension import Extension
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy.distutils.misc_util
import sys, os
import numpy as np

#Check if Cython is installed
try:
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed")
    sys.exit(1)

#TODO: Add branching statement if HMLP_USE_MLI is true
os.environ["CC"] = "mpicc"
os.environ["CXX"] = "mpic++"

#include directories
inc_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
inc_dirs = inc_dirs + ['/workspace/will/dev/hmlp/include']
inc_dirs = inc_dirs + ['/workspace/will/dev/hmlp/gofmm']
inc_dirs = inc_dirs + ['/workspace/will/dev/hmlp/frame']
inc_dirs = inc_dirs + ['/workspace/will/dev/hmlp/frame/base']
inc_dirs = inc_dirs + ['/workspace/will/dev/hmlp/frame/primitives']
inc_dirs = inc_dirs + ['/workspace/will/dev/hmlp/frame/containers']
inc_dirs = inc_dirs + ['/workspace/will/dev/hmlp/kernel/reference']
inc_dirs = inc_dirs + ['/workspace/will/dev/hmlp/kernel/x86_64/haswell']
inc_dirs = inc_dirs + ['/opt/apps/sysnet/intel/17.0/mkl/include']
inc_dirs = inc_dirs + [np.get_include()]

# hmlp library directory
lib_dirs = ['/workspace/will/dev/hmlp/build/lib']
lib_dirs = lib_dirs + ['/opt/apps/sysnet/intel/17.0/mkl/include']
lib_dirs = lib_dirs + ['/opt/apps/sysnet/intel/17.0/mkl/lib/intel64']

def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
        return files

def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension(
        extName,
        [extPath],
        include_dirs = inc_dirs,
        language='c++',
        libraries = ['hmlp'],
        library_dirs = lib_dirs,
        runtime_library_dirs = lib_dirs,
        extra_compile_args=["-std=c++11","-O3","-qopenmp","-m64", "-DUSE_INTEL", "-DUSE_VML", "-DMKL_ILP64", "-mavx", "-DHMLP_USE_MPI", "-I/opt/apps/sysnet/intel/17.0/mkl/include"],
        extra_link_args=["-lpthread","-qopenmp","-mkl=parallel", "-lm", "-L/opt/apps/sysnet/intel/17.0/mkl/lib/intel64/", "-lmkl_rt", "-lpthread", "-lm", "-ldl", "-mavx"]
    )

extNames = scandir("pygofmm")
extensions = [makeExtension(name) for name in extNames]

setup(
    name="pygofmm",
    packages=["pygofmm", "pygofmm.mltools"],
    ext_modules=extensions,
    zip_safe=False,
    include_package_data=True,
    cmdclass = {'build_ext': build_ext}
    )
