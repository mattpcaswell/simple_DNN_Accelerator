#include "ap_fixed.h"

// 4-bit
typedef ap_fixed<4,1,AP_TRN,AP_SAT> activation_t;
typedef ap_fixed<4,1,AP_TRN,AP_SAT> weight_t;
typedef ap_fixed<8,2,AP_TRN,AP_SAT> accumulation_t;

//// 8-bit
//typedef ap_fixed<8,2,AP_TRN,AP_SAT> activation_t;
//typedef ap_fixed<8,2,AP_TRN,AP_SAT> weight_t;
//typedef ap_fixed<16,4,AP_TRN,AP_SAT> accumulation_t;

// 16-bit
//typedef ap_fixed<16,2,AP_TRN,AP_SAT> activation_t;
//typedef ap_fixed<16,2,AP_TRN,AP_SAT> weight_t;
//typedef ap_fixed<32,4,AP_TRN,AP_SAT> accumulation_t;

typedef ap_int<5> exponent_t;

// cifar10 first layer
#define MAX_NUM_FILTERS 32
#define MAX_FILTER_DIM  3

#define MAX_INPUT_SX    32
#define MAX_INPUT_SY    32
#define MAX_INPUT_SZ    3

#define MAX_OUTPUT_SX   32
#define MAX_OUTPUT_SY   32

#define MAX_FILTER_SIZE MAX_FILTER_DIM * MAX_FILTER_DIM * MAX_INPUT_SZ
#define MAX_FILTERS_SIZE MAX_NUM_FILTERS * MAX_FILTER_SIZE
#define MAX_IN_SIZE MAX_INPUT_SX * MAX_INPUT_SY * MAX_INPUT_SZ
#define MAX_OUT_SIZE MAX_OUTPUT_SX * MAX_OUTPUT_SY * MAX_NUM_FILTERS
#define MAX_BIAS_SIZE MAX_NUM_FILTERS

void conv(
		weight_t filters[MAX_NUM_FILTERS][MAX_FILTER_DIM][MAX_FILTER_DIM][MAX_INPUT_SZ],
		exponent_t filter_exponents[MAX_NUM_FILTERS],
		int num_filters,
		int filter_dim_size,
		activation_t in[MAX_INPUT_SX][MAX_INPUT_SY][MAX_INPUT_SZ],
		exponent_t in_exponents[MAX_INPUT_SZ],
		int in_sx,
		int in_sy,
		int in_sz,
		activation_t out[MAX_OUTPUT_SX][MAX_OUTPUT_SY][MAX_NUM_FILTERS],
		exponent_t out_exponents[MAX_NUM_FILTERS],
		int out_sx,
		int out_sy,
		int stride,
		activation_t bias[MAX_BIAS_SIZE]);
