/**
 * Add 2 vectors using CUDA.
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
#define CUDA_CHECK_RETURN( value ) { \
    cudaError_t err = value; \
    if( err != cudaSuccess ) { \
        fprintf( stderr, "Error %s at line %d in file %s\n", \
                        cudaGetErrorString(err), __LINE__, __FILE__ ); \
        exit( 1 ); \
    } }

#define VECT_SIZE (7841u)
#define BLOCK_SIZE (128u)

__global__ void vect_fill( int * data )
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if( i < VECT_SIZE ) data[ i ] = i + 1;
}

__global__ void vect_add( int * vect_1, int * vect_2, int * vect_result )
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if( i < VECT_SIZE ) vect_result[i] = vect_1[i] + vect_2[i];
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char **argv) {
        
    /* Allocate data buffer in host memory for result vector */
    int *h_result = (int*) malloc( VECT_SIZE * sizeof(int) );
    memset( h_result, 0, VECT_SIZE * sizeof(int) );
    
    /* Allocate data buffer in device memory for 3 vectors */
    int *d_vector_1, *d_vector_2, *d_vector_result = NULL;
    CUDA_CHECK_RETURN( cudaMalloc( &d_vector_1, VECT_SIZE * sizeof(int) ) );
    CUDA_CHECK_RETURN( cudaMalloc( &d_vector_2, VECT_SIZE * sizeof(int) ) );
    CUDA_CHECK_RETURN( cudaMalloc( &d_vector_result, VECT_SIZE * sizeof(int) ) );
        
    /* Configure kernel */
    int blockSize = BLOCK_SIZE;
    int gridSize = (VECT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    /* Run kernels to fill 2 vectors */
    vect_fill<<< gridSize, blockSize >>>( d_vector_1 );
    vect_fill<<< gridSize, blockSize >>>( d_vector_2 );
    /* Wait until the kernel finishes its work */
    CUDA_CHECK_RETURN( cudaDeviceSynchronize() );

    /* Add 2 vectors */
    vect_add<<< gridSize, blockSize >>>( d_vector_1, d_vector_2, d_vector_result );
    /* Wait until the kernel finishes its work */
    CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
        
    /* Copy back to host and print */
    CUDA_CHECK_RETURN( cudaMemcpy( h_result, d_vector_result, VECT_SIZE * sizeof(int), cudaMemcpyDeviceToHost) );
    for( unsigned int i = 0; i < VECT_SIZE; ++i ) std::cout << h_result[i] << std::endl;
        
    CUDA_CHECK_RETURN( cudaFree(d_vector_1) );
    CUDA_CHECK_RETURN( cudaFree(d_vector_2) );
    CUDA_CHECK_RETURN( cudaFree(d_vector_result) );
        
    free( h_result );
    
    return 0;
}
