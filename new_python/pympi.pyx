from collections import namedtuple, Iterable

from libcpp cimport bool

cdef extern from "${CMAKE_SOURCE_DIR}/new_python/mpi_stats.hpp":
    bool check_mpi() nogil
    int get_rank() nogil
    int get_comm_size()
    int thread_level()
    bool has_thread_multiple()

MPIStats = namedtuple("MPIStats", ["check_mpi", "rank", "size",
                                   "thread_level", "has_thread_multiple"])

def mpi_stats():
    return MPIStats(check_mpi(), get_rank(), get_comm_size(),
                    thread_level(), has_thread_multiple())
