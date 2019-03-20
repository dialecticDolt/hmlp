#distutils: language = c++
from libcpp cimport bool as bool_t
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
cimport  mpi4py.libmpi as libmpi

cdef extern from "${CMAKE_SOURCE_DIR}/frame/base/DistData.hpp" namespace "hmlp" nogil:

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
        Data() except +
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
        void randn(T mu, T std)
        void randspd(T a, T b)
        bool_t HasIllegalValue()
        void Print()
 
    #Wrapper for Base Distributed Data Class
    cdef cppclass DistDataBase[T](Data[T]):
        DistDataBase(libmpi.MPI_Comm) except +
        DistDataBase(size_t, size_t, int, libmpi.MPI_Comm) except +    
        DistDataBase(size_t, size_t, Data[T]&, libmpi.MPI_Comm) except +
        DistDataBase(size_t, size_t, size_t, size_t, const vector[T], libmpi.MPI_Comm) except +
        DistDataBase(DistDataBase[T]&, libmpi.MPI_Comm) except +        
        #total row and column numbers across all MPI ranks
        size_t row()    
        size_t col()

        #get locally owned row and column numbers
        size_t row_owned()
        size_t col_owned()

        #get comm size and rank
        size_t GetSize()
        size_t GetRank()
    
    #Wrappers for various data distribution types. TODO: Add support for double precision
    #Note: Not all of these are equally developed in Chenhans code. The main ones used are STAR_CBLK/RBKL_STAR 

    cdef cppclass CIRC_CIRC_DistData[T](DistDataBase[T]):
        CIRC_CIRC_DistData(size_t, size_t, int, libmpi.MPI_Comm) except +
        CIRC_CIRC_DistData[T]& operator = (STAR_CIDS_DistData[T]& A) except +    

    cdef cppclass STAR_CBLK_DistData[T](DistDataBase[T]):
        STAR_CBLK_DistData(size_t, size_t, libmpi.MPI_Comm) except +
        STAR_CBLK_DistData(size_t, size_t, T initT, libmpi.MPI_Comm) except +
        #construct from local column data
        STAR_CBLK_DistData(size_t, size_t, Data[T]&, libmpi.MPI_Comm) except +
        STAR_CBLK_DistData(size_t, size_t, vector[T]&, libmpi.MPI_Comm) except +
        #reads column major file
        STAR_CBLK_DistData(size_t, size_t, libmpi.MPI_Comm, string&) except +
        void read(size_t m, size_t n, string&) except +
        
        #operator overloading
        # ()
        T& operator () (size_t, size_t j)
        Data[T]& operator() (vector[size_t]&, vector[size_t]&)
        # = 
        STAR_CBLK_DistData[T]& operator = (const CIRC_CIRC_DistData[T]&) #not implemented in chenhan's code
        STAR_CBLK_DistData[T]& operator = (STAR_CIDS_DistData[T]&) except +

    cdef cppclass RBLK_STAR_DistData[T](DistDataBase[T]):
        RBLK_STAR_DistData() except +
        RBLK_STAR_DistData(size_t, size_t, libmpi.MPI_Comm) except +
        #construct from local row data
        RBLK_STAR_DistData(size_t, size_t, Data[T]&, libmpi.MPI_Comm) except +
        RBLK_STAR_DistData(size_t, size_t, vector[T]&, libmpi.MPI_Comm) except +
        #there are no file read constructors for this distribution in chenhan's code        
        #operator overloading
        #()
        T& operator () (size_t, size_t)
        Data[T]& operator () (vector[size_t]&, vector[size_t]&)
        # = 
        RBLK_STAR_DistData[T]& operator = (const CIRC_CIRC_DistData[T]&) #not implemented in chenhan's code
        RBLK_STAR_DistData[T]& operator = (RIDS_STAR_DistData[T]&) except +

    cdef cppclass STAR_CIDS_DistData[T](DistDataBase[T]):
        STAR_CIDS_DistData(size_t, size_t, vector[size_t]&, libmpi.MPI_Comm) except +
        STAR_CIDS_DistData(size_t, size_t, vector[size_t]&, T initT, libmpi.MPI_Comm) except +
        STAR_CIDS_DistData(size_t, size_t, vector[size_t]&, Data[T]&, libmpi.MPI_Comm) except +

        #operator overloading
        #()
        T& operator () (size_t, size_t)
        Data[T]& operator () (vector[size_t]&, vector[size_t]&)  
        # = 
        STAR_CIDS_DistData[T]& operator = (STAR_CBLK_DistData[T]&) except +
        
        #bookkeeping
        bool_t HasColumn(size_t cid)
        T* columndata(size_t cid)
        pair[size_t, T*] GetIDAndColumnPointer(size_t)
        vector[vector[size_t]] CBLKOwnership() except +
        vector[vector[size_t]] CIRCOwndership(int owner) except +
        

    cdef cppclass STAR_USER_DistData[T](DistDataBase[T]):
        STAR_USER_DistData(size_t, size_t, libmpi.MPI_Comm) except +
        STAR_USER_DistData(STAR_CBLK_DistData[T]&) except +
        void InsertColumns(const vector[size_t]&, Data[T]&) except +
        #operator overloading
        #()
        T& operator () (size_t, size_t)
        Data[T]& operator () (vector[size_t]&, vector[size_t]&)
        
        #bookkeeping
        bool_t HasColumn(size_t cid)
        T* columndata(size_t cid)
        
    cdef cppclass RIDS_STAR_DistData[T](DistDataBase[T]):
        RIDS_STAR_DistData(size_t, size_t, vector[size_t]&, libmpi.MPI_Comm) except +
        RIDS_STAR_DistData(size_t, size_t, string& , libmpi.MPI_Comm) except +
        RIDS_STAR_DistData(RIDS_STAR_DistData[T]&, libmpi.MPI_Comm) except +
        #operator overloading
        #()
        T& operator () (size_t, size_t)
        Data[T]& operator () (vector[size_t]&, vector[size_t]&)
        # = 
        RIDS_STAR_DistData[T]& operator = (RBLK_STAR_DistData[T]&) except +

        #bookkeeping
        vector[vector[size_t]] RBLKOwnership() except +
        
    cdef cppclass STAR_STAR_DistData[T](DistDataBase[T]):
        #This is an empty class in chenhan's code
        STAR_STAR_DistData(size_t, size_t, int, libmpi.MPI_Comm) except +



    #Rework to make DistData templated on datatype(float, double, pair)
    ###################################################
    #cdef cppclass RBLK_STAR_DistData[T]:
    #    RBLK_STAR_DistData(size_t, size_t, int, libmpi.MPI_Comm) except +
    #    #construct from local row data
    #    RBLK_STAR_DistData(size_t, size_t, Data[T]&, libmpi.MPI_Comm) except +
    #    RBLK_STAR_DistData(size_t, size_t, vector[T]&, libmpi.MPI_Comm) except +


    #cdef cppclass STAR_CBLK_DistData[T](DistDataBase[T]):
    #    STAR_CBLK_DistData(size_t, size_t, int, libmpi.MPI_Comm) except +
    #    STAR_CBLK_DistData(size_t, size_t, T initT, libmpi.MPI_Comm) except +
    #    #construct from local column data
    #    STAR_CBLK_DistData(size_t, size_t, Data[T]&, libmpi.MPI_Comm) except +
    #    STAR_CBLK_DistData(size_t, size_t, vector[T]&, libmpi.MPI_Comm) except +
    #    #reads column major file
    #    STAR_CBLK_DistData(size_t, size_t, libmpi.MPI_Comm, string&) except +
    #    void read(size_t m, size_t n, string&) except +
    #    
    #    #operator overloading
    #    # ()
    #    T& operator () (size_t, size_t j)
    #    Data[T]& operator() (vector[size_t]&, vector[size_t]&)
    #    # = 
    #    #STAR_CBLK_DistData& operator = (const CIRC_CIRC_DistData&) #not implemented in chenhan's code
    #    #STAR_CBLK_DistData& operator = (STAR_CIDS_DistData&) except +



    cdef cppclass STAR_CBLK_f_DistData(DistDataBase[float]):
        STAR_CBLK_f_DistData(size_t, size_t, int, libmpi.MPI_Comm) except +
        STAR_CBLK_f_DistData(size_t, size_t, float initT, libmpi.MPI_Comm) except +
        #construct from local column data
        STAR_CBLK_f_DistData(size_t, size_t, Data[float]&, libmpi.MPI_Comm) except +
        STAR_CBLK_f_DistData(size_t, size_t, vector[float]&, libmpi.MPI_Comm) except +
        #reads column major file
        STAR_CBLK_f_DistData(size_t, size_t, libmpi.MPI_Comm, string&) except +
        void read(size_t m, size_t n, string&) except +
        
        #operator overloading
        # ()
        float& operator () (size_t, size_t j)
        Data[float]& operator() (vector[size_t]&, vector[size_t]&)
        # = 
        #STAR_CBLK_f_DistData& operator = (const CIRC_CIRC_f_DistData&) #not implemented in chenhan's code
        #STAR_CBLK_f_DistData& operator = (STAR_CIDS_f_DistData&) except +
