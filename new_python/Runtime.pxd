# distutils: language = c++


cdef extern from "${PROJECT_SOURCE_DIR}/include/hmlp.h":

    ctypedef enum hmlpError_t:
        HMLP_ERROR_SUCCESS,
        HMLP_ERROR_NOT_INITIALIZED,
        HMLP_ERROR_ALLOC_FAILED,
        HMLP_ERROR_INVALID_VALUE,
        HMLP_ERROR_EXECUTION_FAILED,
        HMLP_ERROR_NOT_SUPPORTED,
        HMLP_ERROR_INTERNAL_ERROR

    cdef hmlpError_t hmlp_init() # Does not actually initialize mpi 
    cdef hmlpError_t hmlp_init(int * argc, char *** argv)

    cdef hmlpError_t hmlp_set_num_workers( int n_worker )

    cdef hmlpError_t hmlp_run()

    cdef hmlpError_t hmlp_finalize()
    
