#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "xtime_l.h"
#include "xparameters.h"
#include "xconv.h"
#include "xil_io.h"

// defines / undefines needed to have ap_fixed compile
#undef str
#define HALF_ENABLE_CPP11_CMATH 0
#include "ap_fixed.h"

typedef ap_fixed<16,8,AP_TRN,AP_SAT> activation_t;
typedef ap_fixed<16,8,AP_TRN,AP_SAT> weight_t;

#define INPUT_SX 32
#define INPUT_SY 32
#define INPUT_SZ 3

#define NUM_FILTERS 32
#define FILTER_DIM_SIZE 3
#define FILTER_SZ 3

#define OUTPUT_SX 32
#define OUTPUT_SY 32

#define BIAS_SIZE   NUM_FILTERS
#define INPUT_SIZE  INPUT_SX * INPUT_SY * INPUT_SZ
#define FILTER_SIZE NUM_FILTERS * FILTER_DIM_SIZE * FILTER_DIM_SIZE * FILTER_SZ
#define OUTPUT_SIZE OUTPUT_SX * OUTPUT_SY * NUM_FILTERS

#define STRIDE 1

// HLS HW instance
XConv hlsConv;

int hls_conv_init() {
	XConv_Config *cfgPtr;
	int status;

	cfgPtr = XConv_LookupConfig(XPAR_XCONV_0_DEVICE_ID);
	if (!cfgPtr) {
		print("ERROR: Lookup of conv configuration failed!\n\r");
		return XST_FAILURE;
	}
	status = XConv_CfgInitialize(&hlsConv, cfgPtr);
	if (status != XST_SUCCESS) {
		print("ERROR: Could not initialize conv.\n\r");
		return XST_FAILURE;
	}
	return status;
}

void init_all_blocks() {
	if (hls_conv_init() != XST_SUCCESS) {
		print("HLS conv setup failed!\n\r");
		exit(-1);
	}
}

void fill_bias(weight_t *array) {
	u32 bytes = XConv_Get_bias_V_TotalBytes(&hlsConv);

	XConv_Write_bias_V_Bytes(&hlsConv, 0, (char *) array, bytes);

	print("Finished writing to bias bram!\n");
}

void fill_input(activation_t *array) {
	// split by Z
	unsigned int slice_size = INPUT_SX * INPUT_SY;
	u32 bytes = XConv_Get_in_0_V_TotalBytes(&hlsConv);

	XConv_Write_in_0_V_Bytes(&hlsConv, 0, (char *) &(array[0]), bytes);
	XConv_Write_in_1_V_Bytes(&hlsConv, 0, (char *) &(array[slice_size]), bytes);
	XConv_Write_in_2_V_Bytes(&hlsConv, 0, (char *) &(array[2 * slice_size]), bytes);

	print("Finished writing to input bram!\n");
}

void fill_filters(weight_t *array) {
	// split by filter num
	unsigned int slice_size = FILTER_DIM_SIZE * FILTER_DIM_SIZE * FILTER_SZ;
	u32 bytes = XConv_Get_filters_0_V_TotalBytes(&hlsConv);

	XConv_Write_filters_0_V_Bytes(&hlsConv, 0, (char *) &(array[0]), bytes);
	XConv_Write_filters_1_V_Bytes(&hlsConv, 0, (char *) &(array[1 * slice_size]), bytes);
	XConv_Write_filters_2_V_Bytes(&hlsConv, 0, (char *) &(array[2 * slice_size]), bytes);
	XConv_Write_filters_3_V_Bytes(&hlsConv, 0, (char *) &(array[3 * slice_size]), bytes);
	XConv_Write_filters_4_V_Bytes(&hlsConv, 0, (char *) &(array[4 * slice_size]), bytes);
	XConv_Write_filters_5_V_Bytes(&hlsConv, 0, (char *) &(array[5 * slice_size]), bytes);
	XConv_Write_filters_6_V_Bytes(&hlsConv, 0, (char *) &(array[6 * slice_size]), bytes);
	XConv_Write_filters_7_V_Bytes(&hlsConv, 0, (char *) &(array[7 * slice_size]), bytes);
	XConv_Write_filters_8_V_Bytes(&hlsConv, 0, (char *) &(array[8 * slice_size]), bytes);
	XConv_Write_filters_9_V_Bytes(&hlsConv, 0, (char *) &(array[9 * slice_size]), bytes);
	XConv_Write_filters_10_V_Bytes(&hlsConv, 0, (char *) &(array[10 * slice_size]), bytes);
	XConv_Write_filters_11_V_Bytes(&hlsConv, 0, (char *) &(array[11 * slice_size]), bytes);
	XConv_Write_filters_12_V_Bytes(&hlsConv, 0, (char *) &(array[12 * slice_size]), bytes);
	XConv_Write_filters_13_V_Bytes(&hlsConv, 0, (char *) &(array[13 * slice_size]), bytes);
	XConv_Write_filters_14_V_Bytes(&hlsConv, 0, (char *) &(array[14 * slice_size]), bytes);
	XConv_Write_filters_15_V_Bytes(&hlsConv, 0, (char *) &(array[15 * slice_size]), bytes);
	XConv_Write_filters_16_V_Bytes(&hlsConv, 0, (char *) &(array[16 * slice_size]), bytes);
	XConv_Write_filters_17_V_Bytes(&hlsConv, 0, (char *) &(array[17 * slice_size]), bytes);
	XConv_Write_filters_18_V_Bytes(&hlsConv, 0, (char *) &(array[18 * slice_size]), bytes);
	XConv_Write_filters_19_V_Bytes(&hlsConv, 0, (char *) &(array[19 * slice_size]), bytes);
	XConv_Write_filters_20_V_Bytes(&hlsConv, 0, (char *) &(array[20 * slice_size]), bytes);
	XConv_Write_filters_21_V_Bytes(&hlsConv, 0, (char *) &(array[21 * slice_size]), bytes);
	XConv_Write_filters_22_V_Bytes(&hlsConv, 0, (char *) &(array[22 * slice_size]), bytes);
	XConv_Write_filters_23_V_Bytes(&hlsConv, 0, (char *) &(array[23 * slice_size]), bytes);
	XConv_Write_filters_24_V_Bytes(&hlsConv, 0, (char *) &(array[24 * slice_size]), bytes);
	XConv_Write_filters_25_V_Bytes(&hlsConv, 0, (char *) &(array[25 * slice_size]), bytes);
	XConv_Write_filters_26_V_Bytes(&hlsConv, 0, (char *) &(array[26 * slice_size]), bytes);
	XConv_Write_filters_27_V_Bytes(&hlsConv, 0, (char *) &(array[27 * slice_size]), bytes);
	XConv_Write_filters_28_V_Bytes(&hlsConv, 0, (char *) &(array[28 * slice_size]), bytes);
	XConv_Write_filters_29_V_Bytes(&hlsConv, 0, (char *) &(array[29 * slice_size]), bytes);
	XConv_Write_filters_30_V_Bytes(&hlsConv, 0, (char *) &(array[30 * slice_size]), bytes);
	XConv_Write_filters_31_V_Bytes(&hlsConv, 0, (char *) &(array[31 * slice_size]), bytes);

	print("Finished writing to filter bram!\n");
}

void read_output(activation_t *array) {
	// split by z
	unsigned int slice_size = OUTPUT_SX * OUTPUT_SY;
	u32 bytes = XConv_Get_out_0_V_TotalBytes(&hlsConv);

	XConv_Read_out_0_V_Bytes(&hlsConv, 0, (char *) &(array[0]), bytes);
	XConv_Read_out_1_V_Bytes(&hlsConv, 0, (char *) &(array[1 * slice_size]), bytes);
	XConv_Read_out_2_V_Bytes(&hlsConv, 0, (char *) &(array[2 * slice_size]), bytes);
	XConv_Read_out_3_V_Bytes(&hlsConv, 0, (char *) &(array[3 * slice_size]), bytes);
	XConv_Read_out_4_V_Bytes(&hlsConv, 0, (char *) &(array[4 * slice_size]), bytes);
	XConv_Read_out_5_V_Bytes(&hlsConv, 0, (char *) &(array[5 * slice_size]), bytes);
	XConv_Read_out_6_V_Bytes(&hlsConv, 0, (char *) &(array[6 * slice_size]), bytes);
	XConv_Read_out_7_V_Bytes(&hlsConv, 0, (char *) &(array[7 * slice_size]), bytes);
	XConv_Read_out_8_V_Bytes(&hlsConv, 0, (char *) &(array[8 * slice_size]), bytes);
	XConv_Read_out_9_V_Bytes(&hlsConv, 0, (char *) &(array[9 * slice_size]), bytes);
	XConv_Read_out_10_V_Bytes(&hlsConv, 0, (char *) &(array[10 * slice_size]), bytes);
	XConv_Read_out_11_V_Bytes(&hlsConv, 0, (char *) &(array[11 * slice_size]), bytes);
	XConv_Read_out_12_V_Bytes(&hlsConv, 0, (char *) &(array[12 * slice_size]), bytes);
	XConv_Read_out_13_V_Bytes(&hlsConv, 0, (char *) &(array[13 * slice_size]), bytes);
	XConv_Read_out_14_V_Bytes(&hlsConv, 0, (char *) &(array[14 * slice_size]), bytes);
	XConv_Read_out_15_V_Bytes(&hlsConv, 0, (char *) &(array[15 * slice_size]), bytes);
	XConv_Read_out_16_V_Bytes(&hlsConv, 0, (char *) &(array[16 * slice_size]), bytes);
	XConv_Read_out_17_V_Bytes(&hlsConv, 0, (char *) &(array[17 * slice_size]), bytes);
	XConv_Read_out_18_V_Bytes(&hlsConv, 0, (char *) &(array[18 * slice_size]), bytes);
	XConv_Read_out_19_V_Bytes(&hlsConv, 0, (char *) &(array[19 * slice_size]), bytes);
	XConv_Read_out_20_V_Bytes(&hlsConv, 0, (char *) &(array[20 * slice_size]), bytes);
	XConv_Read_out_21_V_Bytes(&hlsConv, 0, (char *) &(array[21 * slice_size]), bytes);
	XConv_Read_out_22_V_Bytes(&hlsConv, 0, (char *) &(array[22 * slice_size]), bytes);
	XConv_Read_out_23_V_Bytes(&hlsConv, 0, (char *) &(array[23 * slice_size]), bytes);
	XConv_Read_out_24_V_Bytes(&hlsConv, 0, (char *) &(array[24 * slice_size]), bytes);
	XConv_Read_out_25_V_Bytes(&hlsConv, 0, (char *) &(array[25 * slice_size]), bytes);
	XConv_Read_out_26_V_Bytes(&hlsConv, 0, (char *) &(array[26 * slice_size]), bytes);
	XConv_Read_out_27_V_Bytes(&hlsConv, 0, (char *) &(array[27 * slice_size]), bytes);
	XConv_Read_out_28_V_Bytes(&hlsConv, 0, (char *) &(array[28 * slice_size]), bytes);
	XConv_Read_out_29_V_Bytes(&hlsConv, 0, (char *) &(array[29 * slice_size]), bytes);
	XConv_Read_out_30_V_Bytes(&hlsConv, 0, (char *) &(array[30 * slice_size]), bytes);
	XConv_Read_out_31_V_Bytes(&hlsConv, 0, (char *) &(array[31 * slice_size]), bytes);

	print("Finished reading from output bram!\n");
}

int main() {
	// Initialize the system
	print("Starting up\n");
	init_all_blocks();

	print("Setup Complete\n");

	weight_t bias[BIAS_SIZE];
	weight_t filters[FILTER_SIZE];
	activation_t input[INPUT_SIZE];
	activation_t output[OUTPUT_SIZE];

	// Initialize all values to 0
	for (int i = 0; i < BIAS_SIZE; i++)
		bias[i] = weight_t(0.125 * i);
	for (int i = 0; i < FILTER_SIZE; i++)
		filters[i] = 0;
	for (int i = 0; i < INPUT_SIZE; i++)
		input[i] = 0;
	for (int i = 0; i < OUTPUT_SIZE; i++)
		output[i] = 0;

	XTime tStart, tEnd;

	// <-- SET BREAKPOINT HERE FOR LOADING DATA INTO DRAM
	
	// Write data to input, filters, and biases
	fill_bias(bias);
	fill_input(input);
	fill_filters(filters);

	// Perform conv
	XConv_Set_filter_dim_size(&hlsConv, FILTER_DIM_SIZE);
	XConv_Set_in_sx(&hlsConv, INPUT_SX);
	XConv_Set_in_sy(&hlsConv, INPUT_SY);
	XConv_Set_in_sz(&hlsConv, INPUT_SZ);
	XConv_Set_num_filters(&hlsConv, NUM_FILTERS);
	XConv_Set_out_sx(&hlsConv, OUTPUT_SX);
	XConv_Set_out_sy(&hlsConv, OUTPUT_SY);
	XConv_Set_stride(&hlsConv, STRIDE);

	if (XConv_IsReady(&hlsConv)) {
		print("Conv is ready, and staring\n");
	} else {
		print("Conv is not ready. Exiting\n");
		exit(-1);
	}

	// Run Convolution
	XTime_GetTime(&tStart);
	XConv_Start(&hlsConv);
	while(!XConv_IsReady(&hlsConv))
		;
	XTime_GetTime(&tEnd);

	// Read output to DRAM
	read_output(output);

	printf("Conv took %lu clock cycles.\n", 2*(tEnd - tStart));
	printf("Conv took %.6f ms.\n", 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND/1000));
	print("Done!\n");

	return 0;
}
