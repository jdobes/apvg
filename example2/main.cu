/**
 * Simple CUDA application template.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string.h>

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

#define VECT_SIZE (256u)
#define BLOCK_SIZE (128u)

__global__ void vectFill( int * data )
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if( i < VECT_SIZE ) data[ i ] = i + 1;
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char **argv) {
	
	/* Allocate data buffer in host memory */
	int *h_data = (int*) malloc( VECT_SIZE * sizeof(int) );
	memset( h_data, 0, VECT_SIZE * sizeof(int) );
	
	/* Allocate data buffer in device memory */
	int *d_data = NULL;
	CUDA_CHECK_RETURN( cudaMalloc( &d_data, VECT_SIZE * sizeof(int) ) );
	
	/* Configure kernel */
	int blockSize = BLOCK_SIZE;
	int gridSize = (VECT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
	
	/* Run kernel */
	vectFill<<< gridSize, blockSize >>>( d_data );
	
	/* Wait until the kernel finishes its work */
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	
	CUDA_CHECK_RETURN( cudaMemcpy( h_data, d_data, VECT_SIZE * sizeof(int), cudaMemcpyDeviceToHost) );
	
	for( unsigned int i = 0; i < VECT_SIZE; ++i ) std::cout << h_data[i] << std::endl;
	
	CUDA_CHECK_RETURN( cudaFree( d_data) );
	
	free( h_data );
	
	return 0;
}