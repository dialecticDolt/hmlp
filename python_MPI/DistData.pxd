#distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string
cimport  mpi4py.libmpi as libmpi

cdef extern from "${CMAKE_SOURCE_DIR}/frame/base/DistData.hpp" namespace "hmlp":

    ctypedef enum Distribution_t "Distribution_t":
        CBLK "CBLK",
        RBLK "RBLK",
        CIDS "CIDS",
        RIDS "RIDS",
        USER "USER",
        STAR "STAR",
        CIRC "CIRC"
    
    #Wrapper for Base Shared Memory Data Class
    cdef cppclass Data[T]:
        Data(size_t, size_t) except +
        Data(const Data[T]&) except +
        Data(size_t, size_t, const vector[T]) except +        
        Data(size_t, size_t, T*, bool) except +
        
        void read(size_t, size_t, string&) except +
        void write(string&) except +
        void clear()

        size_t row()
        size_t col()
        size_t size()

        void resize()
        T* rowdata(size_t)
        T getvalue(size_t i, size_t j)
        void setvalue(size_t i, size_t j, T v) except +

        void rand(T a, T b)
 
    #Wrapper for Base Distributed Data Class
    cdef cppclass DistDataBase[T](Data[T]):
        DistDataBase(size_t, size_t, int, libmpi.MPI_Comm) except +    
        DistDataBase(size_t, size_t, Data[T]&, libmpi.MPI_Comm) except +
        DistDataBase(size_t, size_t, size_t, size_t, const vector[T], libmpi.MPI_Comm) except +
        
        size_t row()
        size_t col()
        size_t row_owned()
        size_t col_owned()
        size_t GetSize()
        size_t GetRank()
    
    #Wrappers for various data distribution types. TODO: Add support for double precision
    
    #TODO: 
    #       - Add support for the other constructors (local column/row data, from local vector data)
    #       - Add read from file wrapper
    #       - Add wrappers for overloading of () and = 

    cdef cppclass CIRC_CIRC_f_DistData(DistDataBase[float]):
        CIRC_CIRC_f_DistData(size_t, size_t, int, libmpi.MPI_Comm) except +
        #CIRC_CIRC_f_DistData& operator = (CIRC_STAR_f_DistData& A) except +    

    cdef cppclass STAR_CBLK_f_DistData(DistDataBase[float]):
        STAR_CBLK_f_DistData(size_t, size_t, int, libmpi.MPI_Comm) except +

    cdef cppclass RBLK_STAR_f_DistData(DistDataBase[float]):
        RBLK_STAR_f_DistData(size_t, size_t, int, libmpi.MPI_Comm) except +

    cdef cppclass STAR_CIDS_f_DistData(DistDataBase[float]):
        STAR_CIDS_f_DistData(size_t, size_t, int, libmpi.MPI_Comm) except +

    cdef cppclass STAR_USER_f_DistData(DistDataBase[float]):
        STAR_USER_f_DistData(size_t, size_t, int, libmpi.MPI_Comm) except +

    cdef cppclass RIDS_STAR_f_DistData(DistDataBase[float]):
        RIDS_STAR_f_DistData(size_t, size_t, int, libmpi.MPI_Comm) except +

    #cdef cppclass STAR_STAR_f_DistData(DistDataBase[float]):
    #    STAR_STAR_f_DistData(size_t, size_t, int, libmpi.MPI_Comm) except +
        
    
        
        
