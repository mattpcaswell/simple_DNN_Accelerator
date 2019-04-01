#include "ap_fixed.h"

struct point_t
{
	int x, y, z;
};

typedef int data_t;

// 16 bits wide, 8 bit integer part, convergent rounding
//typedef ap_fixed<16,0,AP_TRN> data_t;

#define MAX_FILTERS_SIZE 864
#define MAX_IN_SIZE 3072
#define MAX_OUT_SIZE 32768
#define MAX_BIAS_SIZE 32

void conv(
		data_t filters[MAX_FILTERS_SIZE],
		int num_filters,
		int filter_dim_size,
		data_t in[MAX_IN_SIZE],
		int in_sx,
		int in_sy,
		int in_sz,
		data_t out[MAX_OUT_SIZE],
		int out_sx,
		int out_sy,
		int stride,
		data_t bias[MAX_BIAS_SIZE]);
