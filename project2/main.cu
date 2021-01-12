/**
 * Multiply 2 matrices using CUDA.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string.h>

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

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

#define BLOCK_SIZE (2u)
#define MATRIX_1_WIDTH (6u)
#define MATRIX_1_HEIGHT (4u)
#define MATRIX_2_WIDTH (4u)
#define MATRIX_2_HEIGHT (6u)

__global__ void matrix_fill( Matrix * data )
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if( i < data->width * data->height ) data->elements[i] = i + 1;
}

__global__ void matrix_mul( Matrix * matrix_1, Matrix * matrix_2, Matrix * matrix_result )
{
    float value = 0;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = 0; i < matrix_1->width; ++i) {
        value += matrix_1->elements[row * matrix_1->width + i] * matrix_2->elements[i * matrix_2->width + col];
    }
    matrix_result->elements[row * matrix_result->width + col] = value;
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char **argv) {
        
    /* Allocate data buffer in host memory for result matrix */
    Matrix *h_matrix_result = (Matrix*) malloc( sizeof(Matrix) );
    h_matrix_result->width = MATRIX_2_WIDTH;
    h_matrix_result->height = MATRIX_1_HEIGHT;
    h_matrix_result->elements = (float*) malloc( MATRIX_2_WIDTH * MATRIX_1_HEIGHT * sizeof(float) );
    memset( h_matrix_result->elements, 0, MATRIX_2_WIDTH * MATRIX_1_HEIGHT * sizeof(float) );
    
    /* Allocate data buffer in device memory for 3 matrices */
    Matrix *d_matrix_1, *d_matrix_2, *d_matrix_result = NULL;
    float *d_matrix_1_elements, *d_matrix_2_elements, *d_matrix_result_elements;
    int w, h;

    CUDA_CHECK_RETURN( cudaMalloc( &d_matrix_1, sizeof(Matrix) ) );
    // Cannot allocate in struct in device directly, allocate separately and than link
    CUDA_CHECK_RETURN( cudaMalloc( &d_matrix_1_elements, MATRIX_1_WIDTH * MATRIX_1_HEIGHT * sizeof(float) ) );
    CUDA_CHECK_RETURN( cudaMemcpy( &(d_matrix_1->elements), &d_matrix_1_elements, sizeof(float *), cudaMemcpyHostToDevice) );
    w = MATRIX_1_WIDTH;
    h = MATRIX_1_HEIGHT;
    CUDA_CHECK_RETURN( cudaMemcpy( &(d_matrix_1->width), &w, sizeof(int), cudaMemcpyHostToDevice ) );
    CUDA_CHECK_RETURN( cudaMemcpy( &(d_matrix_1->height), &h, sizeof(int), cudaMemcpyHostToDevice ) );

    CUDA_CHECK_RETURN( cudaMalloc( &d_matrix_2, sizeof(Matrix) ) );
    // Cannot allocate in struct in device directly, allocate separately and than link
    CUDA_CHECK_RETURN( cudaMalloc( &d_matrix_2_elements, MATRIX_2_WIDTH * MATRIX_2_HEIGHT * sizeof(float) ) );
    CUDA_CHECK_RETURN( cudaMemcpy( &(d_matrix_2->elements), &d_matrix_2_elements, sizeof(float *), cudaMemcpyHostToDevice) );
    w = MATRIX_2_WIDTH;
    h = MATRIX_2_HEIGHT;
    CUDA_CHECK_RETURN( cudaMemcpy( &(d_matrix_2->width), &w, sizeof(int), cudaMemcpyHostToDevice ) );
    CUDA_CHECK_RETURN( cudaMemcpy( &(d_matrix_2->height), &h, sizeof(int), cudaMemcpyHostToDevice ) );

    CUDA_CHECK_RETURN( cudaMalloc( &d_matrix_result, sizeof(Matrix) ) );
    // Cannot allocate in struct in device directly, allocate separately and than link
    CUDA_CHECK_RETURN( cudaMalloc( &d_matrix_result_elements, MATRIX_2_WIDTH * MATRIX_1_HEIGHT * sizeof(float) ) );
    CUDA_CHECK_RETURN( cudaMemcpy( &(d_matrix_result->elements), &d_matrix_result_elements, sizeof(float *), cudaMemcpyHostToDevice) );
    w = MATRIX_2_WIDTH;
    h = MATRIX_1_HEIGHT;
    CUDA_CHECK_RETURN( cudaMemcpy( &(d_matrix_result->width), &w, sizeof(int), cudaMemcpyHostToDevice ) );
    CUDA_CHECK_RETURN( cudaMemcpy( &(d_matrix_result->height), &h, sizeof(int), cudaMemcpyHostToDevice ) );
        
    /* Configure kernel */
    int blockSize = BLOCK_SIZE;
    int gridSizeMatrix1 = (MATRIX_1_WIDTH * MATRIX_1_HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int gridSizeMatrix2 = (MATRIX_2_WIDTH * MATRIX_2_HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    /* Run kernels to fill 2 matrices */
    matrix_fill<<< gridSizeMatrix1, blockSize >>>( d_matrix_1 );
    matrix_fill<<< gridSizeMatrix2, blockSize >>>( d_matrix_2 );
    /* Wait until the kernel finishes its work */
    CUDA_CHECK_RETURN( cudaDeviceSynchronize() );

    /* Configure kernel */
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(h_matrix_result->width/dimBlock.x, h_matrix_result->height/dimBlock.y);
    /* Multiply 2 matrices */
    matrix_mul<<< dimGrid, dimBlock >>>( d_matrix_1, d_matrix_2, d_matrix_result );
    /* Wait until the kernel finishes its work */
    CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
        
    /* Copy back to host and print result matrix */
    CUDA_CHECK_RETURN( cudaMemcpy( h_matrix_result->elements, d_matrix_result_elements, MATRIX_2_WIDTH * MATRIX_1_HEIGHT * sizeof(float), cudaMemcpyDeviceToHost) );
    for( int i = 0; i < h_matrix_result->width * h_matrix_result->height; ++i ) {
        std::cout << h_matrix_result->elements[i] << " ";
        
        if ((i+1) % (h_matrix_result->width) == 0) {
            std::cout << std::endl;
        }
    }

    CUDA_CHECK_RETURN( cudaFree(d_matrix_1_elements) );
    CUDA_CHECK_RETURN( cudaFree(d_matrix_1) );
    CUDA_CHECK_RETURN( cudaFree(d_matrix_2_elements) );
    CUDA_CHECK_RETURN( cudaFree(d_matrix_2) );
    CUDA_CHECK_RETURN( cudaFree(d_matrix_result_elements) );
    CUDA_CHECK_RETURN( cudaFree(d_matrix_result) );
        
    free( h_matrix_result->elements );
    free( h_matrix_result );
    
    return 0;
}
