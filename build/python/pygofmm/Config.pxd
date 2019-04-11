# distutils: language = c++
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

#Declaring some useful enums 
cdef extern from "/workspace/will/dev/hmlp/frame/containers/VirtualMatrix.hpp" nogil:
    ctypedef enum DistanceMetric "DistanceMetric":
        GEOMETRY_DISTANCE "GEOMETRY_DISTANCE",
        KERNEL_DISTANCE "KERNEL_DISTANCE", 
        ANGLE_DISTANCE "ANGLE_DISTANCE",
        USER_DISTANCE "USER_DISTANCE"

cdef extern from "/workspace/will/dev/hmlp/include/hmlp.h" nogil:
    ctypedef enum hmlpError_t:
        HMLP_ERROR_SUCCESS,
        HMLP_ERROR_NOT_INITIALIZED,
        HMLP_ERROR_INVALID_VALUE,
        HMLP_ERROR_EXECUTION_FAILED,
        HMLP_ERROR_NOT_SUPPORTED,
        HMMP_ERROR_INTERNAL_ERROR


#Configuration Type: Used as input parameters to gofmm::compress
cdef extern from "/workspace/will/dev/hmlp/gofmm/gofmm.hpp" namespace "gofmm" nogil:
    cdef cppclass Configuration[T]:

        #variables
        DistanceMetric metric_type
        int problem_size
        int neighbor_size
        int maximum_rank
        int maximum_depth
        int leaf_node_size
        float tolerance
        float budget
        bool is_symmetric
        bool use_adaptive_ranks
        bool secure_accuracy

        #constructors
        Configuration() except +
        Configuration(DistanceMetric, int, int, int, int, T, T, bool) except +
        #getters
        DistanceMetric MetricType()
        int ProblemSize()
        int getMaximumDepth()
        int getLeafNodeSize()
        int NeighborSize()
        int MaximumRank()
        T Tolerance()
        T Budget()      
        bool IsSymmetric()
        bool UseAdaptiveRanks()
        bool SecureAccuracy()


        #setters
        hmlpError_t setLeafNodeSize(int)
        hmlpError_t CopyFrom(Configuration&) except +
        hmlpError_t Set(DistanceMetric, int, int, int, int, T, T, bool)
        hmlpError_t setAdaptiveRanks(bool)
        hmlpError_t setSymmetric(bool)
                
