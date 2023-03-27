#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template<const int BLOCKSIZE>
__global__ void sgemm_shared_mem (int M, int N, int K, float alpha, 
                                const float *A, const float *B, float beta, float *C){
    //the current block we are working on
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    //allocate buffer for current block in fast shared mem
    __shared__ float As[BLOCKSIZE*BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE*BLOCKSIZE];

    //index within the block
    const uint threadRow = threadIdx.x / BLOCKSIZE;
    const uint threadCol = threadIdx.x % BLOCKSIZE;

    //advance pointers to the starting pos
    A += cRow * BLOCKSIZE *K; // there are k BLOCK on each row, col = 0
    B += cCol* BLOCKSIZE; // there are cCol BLOCK on each col, row = 0
    C += cRow * BLOCKSIZE*N + cCol *BLOCKSIZE; // row = cRow, col = cCol

    float tmp = 0.0;
    for(int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE){
        //load element in A &B
        // use global index for thread
        As[threadRow*BLOCKSIZE + threadCol] = A[threadRow*K +threadCol];
        Bs[threadRow*BLOCKSIZE + threadCol] = B[threadRow* N+ threadCol];

        // block threads in this block until cache is fully populated
        // ensures that all threads in the same block have reached the same point in the code
        __syncthreads();

        //advance pointers to go to next chunk
        A += BLOCKSIZE;
        B += BLOCKSIZE*N;

        // get the dot product for cached block
        for(int i = 0; i < BLOCKSIZE; ++i){
            tmp += As[threadRow*BLOCKSIZE+i]* Bs[i*BLOCKSIZE+threadCol];
        }

        //sync again at the end to avoid faster threads
        // fetching next block into cache
        __syncthreads();

    }
    C[threadRow*N+threadCol] = alpha*tmp + beta*C[threadRow *N+threadCol];



}