/**
 * Simple CUDA application template.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN( value ) {							\
	cudaError_t err = value;									\
	if( err != cudaSuccess ) {									\
		fprintf( stderr, "Error %s at line %d in file %s\n",	\
				cudaGetErrorString(err), __LINE__, __FILE__ );	\
		exit( 1 );												\
	} }

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char **argv) {
	std::cout << "Hello in CUDA World!" << std::endl;
	
	int devices = 0;
	
	CUDA_CHECK_RETURN( cudaGetDeviceCount( &devices ) );
	
	std::cout << "Available devices: " << devices << std::endl;
	
	cudaDeviceProp properties;
	
	for( int i = 0; i < devices; ++i ) {
		CUDA_CHECK_RETURN( cudaGetDeviceProperties( &properties, i ) );
		std::cout << "Device " << i << " name: " << properties.name << std::endl;
		std::cout << "Compute capability: " << properties.major << "." << properties.minor << std::endl;
		std::cout << "Block dimensions: " << properties.maxThreadsDim[0]
			<< ", " << properties.maxThreadsDim[1]
			<< ", "<< properties.maxThreadsDim[2]
			<< std::endl;
		std::cout << "Grid dimensions: " << properties.maxGridSize[0]
			<< ", " << properties.maxGridSize[1]
			<< ", " << properties.maxGridSize[2]
			<< std::endl;
	}
	
	return 0;
}
