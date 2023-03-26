#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

// A (M*K), B(K*N), C = (M*N)
// alpha and beta are constant in GEMM equation
// A and B are input matrices 
// C is the output matrix
__global__ void sgemm_naive(int M, int N, int K, float alpha, 
                            const float *A, const float *B, float beta, float *C){
    // CUDA thread global index
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint col = blockIdx.y*blockDim.y + threadIdx.y;
    //printf("row: %u col: %u", row,col);
    //printf("threadIdx.x: %u" , threadIdx.x);
    

    if(row < M && col < N){
        float tmp = 0.0;
        for(int i = 0; i < K; ++i){
            tmp += A[row*K+i] * B[i*N+col];
        }

        //GEMM Equation
        // C = α*(A@B)+β*C
        C[row*N+col]= alpha*tmp+ beta*C[row*N+col];
    }

}
