#ifndef AVX_TYPE_H
#define AVX_TYPE_H


/** AVX support */
#include <immintrin.h>
#include <hmlp_internal.hpp>

typedef union 
{
  __m256d v;
  double d[ 4 ];
  __m256i i;
  unsigned long long u[ 4 ];
} v4df_t;

typedef union 
{
  __m128i v;
  int d[ 4 ];
} v4li_t;

#endif /** define AVX_TYPE_H */
