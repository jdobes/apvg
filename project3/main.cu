/**
 * Process images using CUDA.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string.h>
#include "pngio.h"


/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN( value ) { \
    cudaError_t err = value; \
    if( err != cudaSuccess ) { \
        fprintf( stderr, "Error %s at line %d in file %s\n", \
                        cudaGetErrorString(err), __LINE__, __FILE__ ); \
        exit( 1 ); \
    } }

#define FILTER_SIZE (5u)
#define TILE_SIZE (12u) // BLOCK_SIZE - 2 * (FILTER_SIZE / 2)
#define BLOCK_SIZE (16u)

__global__ void processImage( float *filter_matrix, unsigned char * out, const unsigned char * __restrict__ in, size_t pitch, unsigned int width, unsigned int height )
{
    int x_o = TILE_SIZE * blockIdx.x + threadIdx.x;
    int y_o = TILE_SIZE * blockIdx.y + threadIdx.y;
    int x_i = x_o - FILTER_SIZE / 2;
    int y_i = y_o - FILTER_SIZE / 2;
    int sum = 0;

    __shared__ unsigned char sBuffer[BLOCK_SIZE][BLOCK_SIZE];
    
    if( (x_i >= 0) && (x_i < width) && (y_i >= 0) && (y_i < height) )
        sBuffer[threadIdx.y][threadIdx.x] = in[y_i * pitch + x_i];
    else
        sBuffer[threadIdx.y][threadIdx.x] = 0;
    
    __syncthreads();

    if( threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE ) {
        for( int r = 0; r < FILTER_SIZE; ++r ) {
            for( int c = 0; c < FILTER_SIZE; ++c ) {
                sum += sBuffer[threadIdx.y + r][threadIdx.x + c] * filter_matrix[r * FILTER_SIZE + c];
            }
        }
    
        if( x_o < width && y_o < height ) {
            if (sum > 255) {
                sum = 255;
            } else if (sum < 0) {
                sum = 0;
            }
            out[ y_o * width + x_o ] = sum;
        }
    }
}

void create_filter_matrix(char* filter, float* matrix) {
    if (strcmp(filter, "blur") == 0) {
        for (unsigned int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++) {
            matrix[i] = 0;
        }
        matrix[6] = 0.11111;
        matrix[7] = 0.11111;
        matrix[8] = 0.11111;
        matrix[11] = 0.11111;
        matrix[12] = 0.11111;
        matrix[13] = 0.11111;
        matrix[16] = 0.11111;
        matrix[17] = 0.11111;
        matrix[18] = 0.11111;
    } else if (strcmp(filter, "edges") == 0) {
        for (unsigned int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++) {
            matrix[i] = 0;
        }
        matrix[6] = -1;
        matrix[7] = -2;
        matrix[8] = -1;
        matrix[11] = 0;
        matrix[12] = 0;
        matrix[13] = 0;
        matrix[16] = 1;
        matrix[17] = 2;
        matrix[18] = 1;
    } else if (strcmp(filter, "sharpen") == 0) {
        for (unsigned int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++) {
            matrix[i] = 0;
        }
        matrix[6] = 0;
        matrix[7] = -1;
        matrix[8] = 0;
        matrix[11] = -1;
        matrix[12] = 5;
        matrix[13] = -1;
        matrix[16] = 0;
        matrix[17] = -1;
        matrix[18] = 0;
    } else if (strcmp(filter, "emboss") == 0) {
        for (unsigned int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++) {
            matrix[i] = 0;
        }
        matrix[6] = -2;
        matrix[7] = -1;
        matrix[8] = 0;
        matrix[11] = -1;
        matrix[12] = 1;
        matrix[13] = 1;
        matrix[16] = 0;
        matrix[17] = 1;
        matrix[18] = 2;
    } else { // identity
        for (unsigned int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++) {
            matrix[i] = 0;
        }
        matrix[12] = 1;
    }
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char **argv) {

    /* Process args */
    bool args_ok = false;
    bool grayscale = false;
    char *filter, *in_file_name, *out_file_name;
    png_img_t *in_image = NULL;

    if (argc == 4) {
        filter = argv[1];
        in_file_name = argv[2];
        out_file_name = argv[3];
        if ((strcmp(filter, "blur") == 0) || (strcmp(filter, "edges") == 0) || (strcmp(filter, "sharpen") == 0) || (strcmp(filter, "emboss") == 0) || (strcmp(filter, "ident") == 0)) {
            if (strcmp(filter, "edges") == 0) {
                grayscale = true;
            }
            try {
                in_image = new png_img_t(in_file_name);
            } catch (const std::exception &e) {
                std::cout << e.what() << std::endl;
            }

            if (in_image != NULL) {
                std::cout << "Filter: " << filter << std::endl; 
                std::cout << "Input file: " << in_file_name << std::endl;
                std::cout << "Output file: " << out_file_name << std::endl;
                args_ok = true;
            }
        }
    }

    if (!args_ok) {
        std::cout << "Usage: " << argv[0] << " {blur|edges|sharpen|emboss|ident} input.png output.png" << std::endl;
        return 1;
    }

    /* Prepare filter matrix on host */
    float *h_filter_matrix = (float*) malloc( FILTER_SIZE * FILTER_SIZE * sizeof(float) );
    create_filter_matrix(filter, h_filter_matrix);

    /* Copy filter matrix to device */
    float *d_filter_matrix;
    CUDA_CHECK_RETURN( cudaMalloc( &d_filter_matrix, FILTER_SIZE * FILTER_SIZE * sizeof(float) ) );
    CUDA_CHECK_RETURN( cudaMemcpy( d_filter_matrix, h_filter_matrix, FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice) );

    unsigned int width = in_image->get_width();
    unsigned int height = in_image->get_height();
    
    /* Allocate memory buffers for the image processing */
    int size = width * height * sizeof(unsigned char);
    
    /* Allocate image buffers on the host memory */
    unsigned char *h_r = new unsigned char[ size ];
    unsigned char *h_g = new unsigned char[ size ];
    unsigned char *h_b = new unsigned char[ size ];
    
    unsigned char *h_r_n = new unsigned char[ size ];
    unsigned char *h_g_n = new unsigned char[ size ];
    unsigned char *h_b_n = new unsigned char[ size ];
    
    /* Convert PNG image to raw buffer */
    if (grayscale) {
        pvg::pngToRgbGray3( h_r, h_g, h_b, *in_image );
    } else {
        pvg::pngToRgb3( h_r, h_g, h_b, *in_image );
    }
    
    /* Allocate image buffre on GPGPU */
    unsigned char *d_r = NULL;
    unsigned char *d_g = NULL;
    unsigned char *d_b = NULL;
    size_t pitch_r = 0;
    size_t pitch_g = 0;
    size_t pitch_b = 0;
    
    unsigned char *d_r_n = NULL;
    unsigned char *d_g_n = NULL;
    unsigned char *d_b_n = NULL;
    
    CUDA_CHECK_RETURN( cudaMallocPitch( &d_r, &pitch_r, width, height ) );
    CUDA_CHECK_RETURN( cudaMallocPitch( &d_g, &pitch_g, width, height ) );
    CUDA_CHECK_RETURN( cudaMallocPitch( &d_b, &pitch_b, width, height ) );
    
    CUDA_CHECK_RETURN( cudaMalloc( &d_r_n, size ) );
    CUDA_CHECK_RETURN( cudaMalloc( &d_g_n, size ) );
    CUDA_CHECK_RETURN( cudaMalloc( &d_b_n, size ) );    
    
    /* Copy raw buffer from host memory to device memory */
    CUDA_CHECK_RETURN( cudaMemcpy2D( d_r, pitch_r, h_r, width, width, height, cudaMemcpyHostToDevice) );
    CUDA_CHECK_RETURN( cudaMemcpy2D( d_g, pitch_g, h_g, width, width, height, cudaMemcpyHostToDevice) );
    CUDA_CHECK_RETURN( cudaMemcpy2D( d_b, pitch_b, h_b, width, width, height, cudaMemcpyHostToDevice) );
    
    /* Configure image kernel */
    dim3 grid_size( (width + TILE_SIZE - 1) / TILE_SIZE,
                    (height + TILE_SIZE - 1) / TILE_SIZE );
                    
    dim3 block_size( BLOCK_SIZE, BLOCK_SIZE );

    /* Run kernel */
    processImage<<< grid_size, block_size >>>( d_filter_matrix, d_r_n, d_r, pitch_r, width, height );
    processImage<<< grid_size, block_size >>>( d_filter_matrix, d_g_n, d_g, pitch_g, width, height );
    processImage<<< grid_size, block_size >>>( d_filter_matrix, d_b_n, d_b, pitch_b, width, height );
    /* Wait untile the kernel exits */
    CUDA_CHECK_RETURN( cudaDeviceSynchronize() );

    /* Copy raw buffer from device memory to host memory */
	CUDA_CHECK_RETURN( cudaMemcpy( h_r_n, d_r_n, size, cudaMemcpyDeviceToHost ) );
	CUDA_CHECK_RETURN( cudaMemcpy( h_g_n, d_g_n, size, cudaMemcpyDeviceToHost ) );
	CUDA_CHECK_RETURN( cudaMemcpy( h_b_n, d_b_n, size, cudaMemcpyDeviceToHost ) );

    /* Convert raw buffer to PNG image */
    pvg::rgb3ToPng( *in_image, h_r_n, h_g_n, h_b_n );

    /* Write modified image to the disk */
    try {
        in_image->write(out_file_name);
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    /* Free allocated buffers */
    CUDA_CHECK_RETURN( cudaFree( d_r ) );
    CUDA_CHECK_RETURN( cudaFree( d_r_n ) );
    
    CUDA_CHECK_RETURN( cudaFree( d_g ) );
    CUDA_CHECK_RETURN( cudaFree( d_g_n ) );

    CUDA_CHECK_RETURN( cudaFree( d_b ) );
    CUDA_CHECK_RETURN( cudaFree( d_b_n ) );

    delete [] h_r;
    delete [] h_r_n;

    delete [] h_g;
    delete [] h_g_n;

    delete [] h_b;
    delete [] h_b_n;

    free(in_image);

    CUDA_CHECK_RETURN( cudaFree(d_filter_matrix) );
    free(h_filter_matrix);

    return 0;
}
