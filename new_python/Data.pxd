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

        # Copy constructor
        Data( const Data[T]& ) except +
    
        # read from filename
        void read(size_t m , size_t n, string &filename)
        
        # write to filename
        void write(string &filename)

        # clear -- basically the destructor
        void clear()
        
        # get row size
        size_t row()
        
        # get col size
        size_t col()
   
        # get total size
        size_t size()
   
        # resize
        void resize(size_t m, size_t n)
    
    
        # Get entry i,j
        T getvalue(size_t i, size_t j)
    
    
        # set val
        void setvalue(size_t i, size_t j,T v)
    
    
        # randomize uniform on interval [a b]
        void rand(T a, T b)

    
    # end cppclass Data[T]

