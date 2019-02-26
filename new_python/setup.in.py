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
  




from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy.distutils.misc_util
import os
import numpy as np

#TODO: Add branching statement if HMLP_USE_MLI is true
os.environ["CC"] = "mpicc"
os.environ["CXX"] = "mpic++"
os.system("export LD_PRELOAD=${MKLROOT}/lib/intel64/libmkl_core.so:${MKLROOT}/lib/intel64/libmkl_sequential.so")

#include directories
inc_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
inc_dirs = inc_dirs + ['${CMAKE_SOURCE_DIR}/include']
inc_dirs = inc_dirs + ['${CMAKE_SOURCE_DIR}/gofmm']
inc_dirs = inc_dirs + ['${CMAKE_SOURCE_DIR}/frame']
inc_dirs = inc_dirs + ['${CMAKE_SOURCE_DIR}/frame/base']
inc_dirs = inc_dirs + ['${CMAKE_SOURCE_DIR}/frame/primitives']
inc_dirs = inc_dirs + ['${CMAKE_SOURCE_DIR}/frame/containers']
inc_dirs = inc_dirs + ['${CMAKE_SOURCE_DIR}/kernel/reference']
inc_dirs = inc_dirs + ['${CMAKE_SOURCE_DIR}/kernel/${HMLP_ARCH}']
inc_dirs = inc_dirs + ['${MKLROOT}/include']
inc_dirs = inc_dirs + [np.get_include()]
print(inc_dirs)

# hmlp library directory
lib_dirs = ['${CMAKE_BINARY_DIR}/lib']
lib_dirs = lib_dirs + ['${MPI_CXX_LIBRARIES}']
lib_dirs = lib_dirs + ['${MKLROOT}/include']
lib_dirs = lib_dirs + ['${MKLROOT}/lib/intel64']
#lib_dirs = lib_dirs + ['${MKLROOT}/lib/intel64/libmkl_core.so']
#lib_dirs = lib_dirs + ['${MKLROOT}/lib/intel64/libmkl_mkl_intel_thread.so']
print(lib_dirs)

extension_mod_matrix = Extension( 
  "PyGOFMM", 
  sources = ['${CMAKE_BINARY_DIR}/new_python/PyGOFMM.pyx'],
  language="c++",
  include_dirs = inc_dirs,
  libraries = ['hmlp'],
  library_dirs = lib_dirs,
  runtime_library_dirs = lib_dirs,
  extra_compile_args=["${HMLP_PYTHON_CFLAGS}", "-DUSE_INTEL", "-DUSE_VML", "-DMKL_ILP64", "-mavx", "-DHMLP_USE_MPI", "-I${MKLROOT}/include"],
  #extra_compile_args=["-fopenmp", "-O3", "-std=c++11",
  #	"-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"],
  extra_link_args=["${HMLP_PYTHON_LINKER_FLAGS}", "-L${MKLROOT}/lib/intel64/", "-lmkl_rt", "-lpthread", "-lm", "-ldl", "-mavx"]
  #extra_link_args=["${HMLP_PYTHON_LINKER_FLAGS}", "-Wl,--start-group", "${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a", "${MKLROOT}/lib/intel64/libmkl_intel_thread.a", "-Wl,--end-group", "-ldl", "-liomp5", "-lpthread", "-lmkl_rt", "-mavx"]
)

setup(
  ext_modules = cythonize([extension_mod_matrix]) )
