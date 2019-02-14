# distutils: language = c++
from libcpp.vector cimport vector
from libcpp.string cimport string


cdef extern from "${CMAKE_SOURCE_DIR}/frame/base/Data.hpp" namespace "hmlp":
    cdef cppclass Data[T,ALLOCATOR=*]:
        ctypedef T value_type
        ctypedef ALLOCATOR allocator_type

        # default constructor
        Data() except +

        # initialize by size
        Data(size_t m,size_t n) except + 
    
        # initialize by filename
        #Data(size_t, size_t, string) except +

        # copy constructor -- TODO


        # write to file --TODO
   
   
        # resize
        void resize(size_t m, size_t n)
    
    
        # Get entry i,j
        #T getvalue(size_t i, size_t j)
    
    
        # set val -- TODO how to handle templates??
        #void setvalue(size_t i, size_t j,T v)
    
    
        # randomize
        #void rand()

    
    # end cppclass Data[T]

