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


#define MAX_NUM_FILTERS 3
#define MAX_FILTER_DIM  3

#define MAX_INPUT_SX    16
#define MAX_INPUT_SY    16
#define MAX_INPUT_SZ    3

#define MAX_OUTPUT_SX   16
#define MAX_OUTPUT_SY   16

#define MAX_FILTER_SIZE MAX_FILTER_DIM * MAX_FILTER_DIM * MAX_INPUT_SZ
#define MAX_FILTERS_SIZE MAX_NUM_FILTERS * MAX_FILTER_SIZE
#define MAX_IN_SIZE MAX_INPUT_SX * MAX_INPUT_SY * MAX_INPUT_SZ
#define MAX_OUT_SIZE MAX_OUTPUT_SX * MAX_OUTPUT_SY * MAX_INPUT_SZ
#define MAX_BIAS_SIZE MAX_NUM_FILTERS

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
