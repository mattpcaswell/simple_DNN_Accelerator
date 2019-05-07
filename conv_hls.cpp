#include "conv_hls.h"
#include <assert.h>
using namespace std;

/*
int calc_filter_index(int filter_num, int x, int y, int z, int filter_size, int filter_sz) {
#pragma HLS INLINE
	assert(x >= 0 && x < filter_size);
	assert(y >= 0 && y < filter_size);
	assert(z >= 0 && z < filter_sz);

	int i = filter_num * filter_size * filter_size * filter_sz; // get to the right filter
	i += z * (filter_size * filter_size);
	i += x * (filter_size);
	i += y;

	return i;
}

int calc_index(int x, int y, int z, int sx, int sy) {
#pragma HLS INLINE
	assert(x >= 0 && x < sx);
	assert(y >= 0 && y < sy);

	int i = z * sx * sy;
	i += x * sy;
	i += y;

	return i;
}
*/

void conv(weight_t filters[MAX_NUM_FILTERS][MAX_FILTER_DIM][MAX_FILTER_DIM][MAX_INPUT_SZ], int num_filters,
		int filter_dim_size, activation_t in[MAX_INPUT_SX][MAX_INPUT_SY][MAX_INPUT_SZ], int in_sx, int in_sy,
		int in_sz, activation_t out[MAX_OUTPUT_SX][MAX_OUTPUT_SY][MAX_NUM_FILTERS], int out_sx, int out_sy,
		int stride, activation_t bias[MAX_BIAS_SIZE]) {

#pragma HLS RESOURCE variable=filters core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=in core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=out core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=bias core=RAM_1P_BRAM

#pragma HLS ARRAY_PARTITION variable=in complete dim=3
#pragma HLS ARRAY_PARTITION variable=out complete dim=3
#pragma HLS ARRAY_PARTITION variable=filters complete dim=4
#pragma HLS INTERFACE bram port=filters
#pragma HLS INTERFACE bram port=in
#pragma HLS INTERFACE bram port=out
#pragma HLS INTERFACE bram port=bias

#pragma HLS INTERFACE s_axilite bundle=ctrl port=return
#pragma HLS INTERFACE s_axilite bundle=ctrl port=num_filters
#pragma HLS INTERFACE s_axilite bundle=ctrl port=filter_dim_size
#pragma HLS INTERFACE s_axilite bundle=ctrl port=in_sx
#pragma HLS INTERFACE s_axilite bundle=ctrl port=in_sy
#pragma HLS INTERFACE s_axilite bundle=ctrl port=in_sz
#pragma HLS INTERFACE s_axilite bundle=ctrl port=out_sx
#pragma HLS INTERFACE s_axilite bundle=ctrl port=out_sy
#pragma HLS INTERFACE s_axilite bundle=ctrl port=stride

	assert(num_filters <= MAX_NUM_FILTERS);
	assert(filter_dim_size <= MAX_FILTER_DIM);
	assert(in_sx <= MAX_INPUT_SX);
	assert(in_sy <= MAX_INPUT_SY);
	assert(in_sz <= MAX_INPUT_SZ);
	assert(out_sx <= MAX_OUTPUT_SX);
	assert(out_sy <= MAX_OUTPUT_SY);

	// Zero out the output
	for (int x = 0; x < out_sx; x++) {
		for (int y = 0; y < out_sy; y++) {
			for (int z = 0; z < num_filters; z++) {
				out[x][y][z] = bias[z];
			}
		}
	}

	int needed_in_sx = out_sx * stride + filter_dim_size - 1;
	int needed_in_sy = out_sy * stride + filter_dim_size - 1;

	// Calculate zero padding. Evenly distribute with the 1 extra going on the right/bottom. Equivalent to the "same" padding in Keras
	int padx = (needed_in_sx - in_sx) / 2;
	int pady = (needed_in_sy - in_sy) / 2;

	int current_location_x, current_location_y;
	for (int i = 0; i < MAX_FILTER_DIM; i++) {    										// filter x (K)
		if (i < filter_dim_size) {
			for (int j = 0; j < MAX_FILTER_DIM; j++) {									// filter y (K)
				if (j < filter_dim_size) {
					for (int x = 0; x < MAX_OUTPUT_SX; x++) {     						// output x (R)
						if (x < out_sx) {
							for (int y = 0; y < MAX_OUTPUT_SY; y++) { 					// output y (C)
#pragma HLS PIPELINE
								if (y < out_sy) {
									for (int z = 0; z < MAX_NUM_FILTERS; z++) {			// output z (M)
#pragma HLS DEPENDENCE variable=out inter false
#pragma HLS UNROLL
										if (z < num_filters) {
											accumulation_t sum = 0;

											for (int k = 0; k < MAX_INPUT_SZ; k++) {	// filter z (N)
#pragma HLS UNROLL
												if (k < in_sz) {
													current_location_x = (x * stride) + i - padx;
													current_location_y = (y * stride) + j - pady;
													// Keep v at 0 if the location is outside of the input AKA zero padding
													if (current_location_x >= 0
															&& current_location_x < in_sx
															&& current_location_y >= 0
															&& current_location_y < in_sy) {
														weight_t f = filters[z][i][j][k];
														activation_t v = in[current_location_x][current_location_y][k];
														sum += v * f;
													}
												}
											}

											out[x][y][z] += sum / accumulation_t(256);
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}
