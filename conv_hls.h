#include "ap_fixed.h"

struct point_t
{
	int x, y, z;
};

//typedef int data_t;

//typedef ap_int<8> data_t;

// 16 bits wide, 8 bit integer part, convergent rounding
typedef ap_fixed<32,32,AP_TRN,AP_SAT> activation_t;
typedef ap_fixed<32,32,AP_TRN,AP_SAT> weight_t;
typedef ap_fixed<32,32,AP_TRN,AP_SAT> accumulation_t;

#define MAX_FILTERS_SIZE 864
#define MAX_IN_SIZE 3072
#define MAX_OUT_SIZE 32768
#define MAX_BIAS_SIZE 32

void conv(
		weight_t filters[MAX_FILTERS_SIZE],
		int num_filters,
		int filter_dim_size,
		activation_t in[MAX_IN_SIZE],
		int in_sx,
		int in_sy,
		int in_sz,
		activation_t out[MAX_OUT_SIZE],
		int out_sx,
		int out_sy,
		int stride,
		activation_t bias[MAX_BIAS_SIZE]);
