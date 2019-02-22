from Runtime cimport *



cdef class PyRuntime:
    cpdef int isInit

    def __cinit__( self ):
        self.isInit = int(0)

    def __dealloc__( self ):
        hmlp_finalize()

    cpdef init( self ):
        hmlp_init()
        self.isInit = int(1)

    cpdef set_num_workers( self, int nworkers ):
        hmlp_set_num_workers( nworkers )

    cpdef run( self ):
        hmlp_run()

    cpdef finalize( self ):
        hmlp_finalize()
