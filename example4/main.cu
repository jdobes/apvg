/**
 * Simple CUDA application template.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include "pngio.h"

#define WIDTH (800u)
#define HEIGHT (600u)
#define MAX_ITER (7650u)

#define BLOCK_SIZE (16u)

#define USE_GPU 1

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
	
__host__ __device__ void calcMandelbrotPixel( unsigned char *img, unsigned int x, unsigned int y )
{
	double x0, y0, xb, yb, xtmp;
	unsigned int i, iter;
	
	if( x < WIDTH && y < HEIGHT ) {
		i = ( x + y * WIDTH ) * 3;
		
		x0 = ( (double)x / WIDTH * 3.5 ) - 2.5;
		y0 = ( (double)y / HEIGHT * 2 ) - 1;

		xb = 0;
		yb = 0;

		iter = 0;

		while( xb*xb + yb*yb < 4 && iter < MAX_ITER ) {
			xtmp = xb*xb - yb*yb + x0;
			yb = 2*xb*yb + y0;

			xb = xtmp;
			iter++;
		}
				
		iter*=20;
		
		img[ i ] = iter > 510 ? ( iter - 510 ) % 255 : 0;
		img[ i + 1 ] = iter > 255 ? ( iter - 255 ) % 255 : 0;
		img[ i + 2 ] = iter % 255;
	}
}

__global__ void createImgGPU( unsigned char *img )
{
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	calcMandelbrotPixel( img, x, y );
}

void createImgCPU( unsigned char *img )
{
	for( unsigned int y = 0; y < HEIGHT; ++y )
		for( unsigned x = 0; x < WIDTH; ++x )
			calcMandelbrotPixel( img, x, y );
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char **argv) {
	std::cout << "Processing image..." << std::endl;
	
	png::image<png::rgb_pixel> img( WIDTH, HEIGHT );
	
	int size = WIDTH * 3 * HEIGHT * sizeof(unsigned char);
	
	/* Allocate image buffer on the host memory */
	unsigned char *h_img = new unsigned char[ size ];
	
	/* Allocate image buffre on GPGPU */
	unsigned char *d_img = NULL;
	CUDA_CHECK_RETURN( cudaMalloc( &d_img, size ) );
	
	/* Configure image kernel */
	dim3 grid_size( (WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE,
					(HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE );
					
	dim3 block_size( BLOCK_SIZE, BLOCK_SIZE );
	
	#if USE_GPU
	
	double start = omp_get_wtime();
	createImgGPU<<< grid_size, block_size >>>( d_img );
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	double end = omp_get_wtime();
	
	CUDA_CHECK_RETURN( cudaMemcpy( h_img, d_img, size, cudaMemcpyDeviceToHost ) );
	
	#else
	
	double start = omp_get_wtime();
	createImgCPU( h_img );
	double end = omp_get_wtime();
	
	#endif //USE_GPU
	
	pvg::rgbToPng( img, h_img );
	
	std::cout << "Done in " << end - start << " seconds." << std::endl;
	
	img.write("../output.png");
	
	CUDA_CHECK_RETURN( cudaFree( d_img ) );
	delete [] h_img;
	
	return 0;
}