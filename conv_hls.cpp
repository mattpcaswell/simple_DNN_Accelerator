#include "conv_hls.h"
#include <assert.h>
using namespace std;

point_t map_to_input(point_t out, int z, int stride) {
	out.x *= stride;
	out.y *= stride;
	out.z = z;
	return out;
}

int calc_filter_index(int filter_num, int x, int y, int z, int filter_size) {
	assert(x >= 0 && x < filter_size);
	assert(y >= 0 && y < filter_size);
	assert(z >= 0 && z < filter_size);

	int i = filter_num * filter_size * filter_size * filter_size; // get to the right filter
	i += z * (filter_size * filter_size);
	i += x * (filter_size);
	i += y;

	return i;
}

int calc_index(int x, int y, int z, int sx, int sy) {
	assert(x >= 0 && x < sx);
	assert(y >= 0 && y < sy);

	int i = z * sx * sy;
	i += x * sy;
	i += y;

	return i;
}

void conv(weight_t filters[MAX_FILTERS_SIZE], int num_filters,
		int filter_dim_size, activation_t in[MAX_IN_SIZE], int in_sx, int in_sy,
		int in_sz, activation_t out[MAX_OUT_SIZE], int out_sx, int out_sy,
		int stride, activation_t bias[MAX_BIAS_SIZE]) {
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

	int needed_in_sx = out_sx * stride + filter_dim_size - 1;
	int needed_in_sy = out_sy * stride + filter_dim_size - 1;

	// Calculate zero padding. Evenly distribute with the 1 extra going on the right/bottom. Equivalent to the "same" padding in Keras
	int padx = (needed_in_sx - in_sx) / 2;
	int pady = (needed_in_sy - in_sy) / 2;

	int current_location_x, current_location_y;
	for (int filter = 0; filter < num_filters; filter++) { 		// filter
		for (int x = 0; x < out_sx; x++) {     					// output x
			for (int y = 0; y < out_sy; y++) { 					// output y
				point_t input = { x, y, 0 };
				point_t mapped = map_to_input(input, 0, stride);
				accumulation_t sum = 0;
				for (int i = 0; i < MAX_FILTER_DIM; i++)     	// filter x
#pragma HLS PIPELINE
					if (i < filter_dim_size) {
						for (int j = 0; j < MAX_FILTER_DIM; j++)	// filter y
#pragma HLS UNROLL
							if (j < filter_dim_size) {
								for (int z = 0; z < MAX_INPUT_SZ; z++) { // filter z
#pragma HLS UNROLL
									if (z < in_sz) {
										weight_t f = filters[calc_filter_index(filter, i, j, z, filter_dim_size)];
										activation_t v = 0;
										
										current_location_x = mapped.x + i - padx;
										current_location_y = mapped.y + j - pady;
										// Keep v at 0 if the location is outside of the input AKA zero padding
										if (current_location_x >= 0
												&& current_location_x < in_sx
												&& current_location_y >= 0
												&& current_location_y < in_sy)
											v = in[calc_index(current_location_x, current_location_y, z, in_sx, in_sy)];

										sum += f * v;
									}
								}
							}
					}
				accumulation_t result = sum / accumulation_t(256); // 2^8 = 256
				result = result + bias[filter];
				out[calc_index(x, y, filter, out_sx, out_sy)] = result;
			}
		}
	}
}
