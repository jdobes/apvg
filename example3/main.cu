/**
 * Simple CUDA application template.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <omp.h>

#include "pngio.h"

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
	
#define WIDTH (800u)
#define HEIGHT (600u)
#define BLOCK_SIZE (16u)

double h_diag_len = 0.0;
/* Variable in constant GPU memory */
__constant__ double d_diag_len;

__global__ void createImg( unsigned char * img )
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if( x < WIDTH && y < HEIGHT ) {
		unsigned int i = (y * WIDTH + x) * 3;
		
		img[ i ] = float(x) / WIDTH * 255;
		img[ i + 1 ] = float(y) / HEIGHT * 255;
		img[ i + 2 ] = sqrtf( powf(x, 2) + powf(y, 2) ) / d_diag_len * 255;
	}
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char **argv) {
	std::cout << "Creating image, please wait..." << std::endl;
	
	int size = WIDTH * HEIGHT * 3 * sizeof( unsigned char );
	h_diag_len = sqrtf( powf( WIDTH, 2) + powf( HEIGHT, 2) );
	
	/* Initialize constant variable in device memory */
	CUDA_CHECK_RETURN( cudaMemcpyToSymbol( d_diag_len, &h_diag_len, sizeof(double) ) );
	
	/* Allocated memory buffer in host memory */
	unsigned char * h_img = new unsigned char[ size ];
	
	/* Allocate memory buffer in device memory */
	unsigned char * d_img;
	CUDA_CHECK_RETURN( cudaMalloc( &d_img, size ) );
	
	/* Kernel configuration */
	dim3 blockSize( BLOCK_SIZE, BLOCK_SIZE );
	dim3 gridSize( (WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE,
				   (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE );
	
	/* Run kernel */
	double start = omp_get_wtime();
	createImg<<<gridSize, blockSize>>>( d_img );
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	double end = omp_get_wtime();
	
	/* Copy device memory buffer to the host memory */
	CUDA_CHECK_RETURN( cudaMemcpy( h_img, d_img, size, cudaMemcpyDeviceToHost ) );
	
	/* Create PNG image */
	png::image<png::rgb_pixel> img( WIDTH, HEIGHT );
	pvg::rgbToPng( img, h_img );
	
	/* Write the image to disk */
	img.write( "../output.png" );
	
	std::cout << "Done in " << end - start << " seconds." << std::endl;
	
	/* Free memory */
	delete [] h_img;
	CUDA_CHECK_RETURN( cudaFree( d_img ) );
	
	return 0;
}