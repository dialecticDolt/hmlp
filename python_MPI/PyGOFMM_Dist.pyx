from DistData cimport *
cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi


cdef class PyData:
    cdef DistData<DTYPE, DTYPE,float>* c_data

    def __cinit__(self, size_t m, size_t n, int owner, MPI.Comm comm):
        self.c_data = new DistData<DTYPE, DTYPE, float>[float](m, n, owner, comm.ob_mpi)
