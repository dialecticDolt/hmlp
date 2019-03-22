#distutils: language =c++
cimport numpy as np
#from DistMatrix cimport * 
from libcpp.vector cimport vector
#from libc.math cimport max

cdef extern from "${CMAKE_SOURCE_DIR}/new_python/custom/user_defined.cpp" nogil:
    cdef T custom_element_kernel[T, TP](const void* param, const TP* x, const TP* y, size_t d) nogil
    cdef void custom_matrix_kernel[T, TP](const void* param, const TP* X, const TP*Y, size_t d, T* K, size_t m, size_t n) nogil


#cdef inline float custom_element_kernel(const void* param, const float* x, const float* y, size_t d):
#     cdef float a;
#     a = 2.0;
#     a = kernel_s[float, float].innerProduct(x, y, d);
#     #print("x py-address::", <unsigned long>x)
#     #print("y py-address::", <unsigned long>y)  
#     return max(float(0), a)

#cdef inline void custom_matrix_kernel(const void* param, const float* X, const float*Y, size_t d, float* K, size_t m, size_t n):
#    kernel_s[float, float].innerProducts(X, Y, d, K, m, n)
#    #print("X py-address::", <unsigned long>X)
#    #print("Y py-address::", <unsigned long>Y) 
#    print(<int>m*n)
#    print(<float>K[1])
#    print(<float>K[2])
#    for i in range(0, m*n):
#        K[i] = max(float(0), K[i])
    
