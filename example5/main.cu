/**
 * Simple CUDA application template.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include "pngio.h"

#define BLOCK_SIZE (16u)

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

__global__ void processImage( unsigned char * img, unsigned int width, unsigned int height )
{
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if( x < width && y < height ) {
		int i = ( x + y * width ) * 3;
		
		unsigned int r = img[ i + 2 ];
		
		/* Replace blue pixels with yellow pixels */
		if( r > 0 ) {
			img[ i ] = r;
			img[ i + 1 ] = r;
			img[ i + 2 ] = 0;
		}
	}
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char **argv) {
	printf("Processing image. See 'lena_new.png' image for the results...\n");
	
	/* Load image from file */
	png::image<png::rgb_pixel> img("../lena.png");
	
	unsigned int width = img.get_width();
	unsigned int height = img.get_height();
	
	/* Allocate memory buffers for the image processing */
	int size = width * 3 * height * sizeof(unsigned char);
	
	/* Allocate image buffer on the host memory */
	unsigned char *h_img = new unsigned char[ size ];
	
	/* Convert PNG image to raw buffer */
	pvg::pngToRgb( h_img, img );
	
	/* Allocate image buffre on GPGPU */
	unsigned char *d_img = NULL;
	CUDA_CHECK_RETURN( cudaMalloc( &d_img, size ) );
	
	/* Copy raw buffer from host memory to device memory */
	CUDA_CHECK_RETURN( cudaMemcpy( d_img, h_img, size, cudaMemcpyHostToDevice) );
	
	/* Configure image kernel */
	dim3 grid_size( (width + BLOCK_SIZE - 1) / BLOCK_SIZE,
					(height + BLOCK_SIZE - 1) / BLOCK_SIZE );
					
	dim3 block_size( BLOCK_SIZE, BLOCK_SIZE );
	
	/* Run kernel and measure processing time */
	double start = omp_get_wtime();
	processImage<<< grid_size, block_size >>>( d_img, width, height );
	/* Wait untile the kernel exits */
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	double end = omp_get_wtime();
	
	/* Copy raw buffer from device memory to host memory */
	CUDA_CHECK_RETURN( cudaMemcpy( h_img, d_img, size, cudaMemcpyDeviceToHost ) );
	
	/* Convert raw buffer to PNG image */
	pvg::rgbToPng( img, h_img );
	
	std::cout << "Done in " << end - start << " seconds." << std::endl;
	
	/* Write modified image to the disk */
	img.write("../lena_new.png");
	
	/* Free allocated buffers */
	CUDA_CHECK_RETURN( cudaFree( d_img ) );
	delete [] h_img;
	
	return 0;
}