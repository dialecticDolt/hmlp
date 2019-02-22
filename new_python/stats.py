# Make sure to call this first to initialize MPI
import mpi4py
mpi4py.rc=True
mpi4py.rc='multiple'
from mpi4py import MPI
from pympi import *

print(mpi_stats())
