#include "conv_hls.h"
#include <fstream>
#include <iostream>

int main(int argc, char** args) {
	data_t filters[MAX_FILTERS_SIZE];
	data_t bias[MAX_BIAS_SIZE];
	data_t in[MAX_IN_SIZE];
	data_t out[MAX_OUT_SIZE];

	int num_filters = 32;
	int filter_dim_size = 3;
	int in_sx = 32;
	int in_sy = 32;
	int in_sz = 3;
	int out_sx = 32;
	int out_sy = 32;
	int stride = 1;

	std::ifstream weights("/home/vlsi_lab/Desktop/Quick_test/for_hardware/weights.mem");
	std::ifstream biases("/home/vlsi_lab/Desktop/Quick_test/for_hardware/biases.mem");
	std::ifstream image("/home/vlsi_lab/Desktop/Quick_test/for_hardware/image.mem");

	// read in filters
	int a;
	for (int i = 0; i < MAX_FILTERS_SIZE; i++) {

		if (!(weights >> a)) {
			std::cout << "Weights file too small!\n";
			exit(-1);
		}
		filters[i] = a;
	}
	if ((weights >> a)) {
		std::cout << "Weights file too large!\n";
		exit(-1);
	}

	// read in bias
	for (int i = 0; i < MAX_BIAS_SIZE; i++) {
		if (!(biases >> a)) {
			std::cout << "Biases file too small!\n";
			exit(-1);
		}
		bias[i] = a;
	}
	if ((biases >> a)) {
		std::cout << "Biases file too large!\n";
		exit(-1);
	}

	// read in input image
	for (int i = 0; i < MAX_IN_SIZE; i++) {
		if (!(image >> a)) {
			std::cout << "Image file too small!\n";
			exit(-1);
		}
		in[i] = a;
	}
	if ((image >> a)) {
		std::cout << "Image file too large!\n";
		exit(-1);
	}

	conv(filters, num_filters, filter_dim_size, in, in_sx, in_sy, in_sz, out, out_sx, out_sy, stride, bias);

	std::ofstream output("/home/vlsi_lab/Desktop/Quick_test/for_hardware/output-hls.mem");
	for (int i = 0; i < MAX_OUT_SIZE; i++) {
		output << out[i] << "\n";
	}
}
