#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// size of the block (how many threads are we taking to calculation each time)
template<const uint BLOCKSIZE>
__global__ void sgemm_mem_coalesce(int M, int N, int K, float alpha, 
                                    const float *A, const float *B, float beta, float *C){
    const uint row = blockIdx.x * BLOCKSIZE +(threadIdx.x/BLOCKSIZE);
    const uint col = blockIdx.y* BLOCKSIZE +(threadIdx.x%BLOCKSIZE);

    if(row < M && col < N){
        float tmp = 0.0;
        for(int i = 0; i < K; ++i){
            tmp += A[row*K +i] * B[i*N+col];
        }

        C[row*N+col] = alpha*tmp+beta*C[row*N+col];
    }

}