from Runtime cimport *
from cython.operator cimport dereference as deref

cdef extern from "Python.h":
    char* PyUnicode_AsUTF8(object unicode)

from libc.stdlib cimport malloc, free
from libc.string cimport strcmp

cdef char ** to_cstring_array(list_str):
    cdef char **ret = <char **>malloc(len(list_str) * sizeof(char *))
    for i in xrange(len(list_str)):
        ret[i] = PyUnicode_AsUTF8(list_str[i])
    return ret

cdef class PyRuntime:
    cpdef int isInit

    def __cinit__( self ):
        self.isInit = int(0)

    def __dealloc__( self ):
        hmlp_finalize()

    cpdef init( self ): 
        # create dummy args -- doesn't work properly with none
        cdef int arg_c = 7
        cdef char **arg_v = <char **>malloc(7 * sizeof(char *))
        #cdef char** arg_v = to_cstring_array(str_list)
        hmlp_init(&arg_c, &arg_v)
        #self.isInit = int(1)

    cpdef set_num_workers( self, int nworkers ):
        if self.isInit is 1:
            hmlp_set_num_workers( nworkers )

    cpdef run( self ):
        hmlp_run()

    cpdef finalize( self ):
        hmlp_finalize()
