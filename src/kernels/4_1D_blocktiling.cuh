#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template<const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_1D_blocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C){
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    //each warp will calculate 32*TM elements
    const int threadCol = threadIdx.x %BN;
    const int threadRow = threadIdx.x /BN;

    //allocate space for the current blocktile
    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];

    //Move block to beginning of A's row and B's row
    A += cRow * BM *K;
    B += cCol *BN;
    C += cRow * BM* N + cCol*BN;

    //TODO: what are those two lines of code
    assert(BM*BK == blockDim.x);
    assert(BN * BK == blockDim.x);

    const uint innerColA = threadIdx.x %BK;
    const uint innerRowA = threadIdx.x/BK;

    const uint innerColB = threadIdx.x %BN;
    const uint innerRowB = threadIdx.x /BN;

    //allocate thread-local cache for results in registerfile
    float threadResults[TM] = {0.0};

    for(uint bkIdx = 0; bkIdx < K; bkIdx += BK){
        As[innerRowA* BK +innerColA] = A[innerRowA*K +innerColA];
        Bs[innerRowB* BN + innerColB] = B[innerRowB*N + innerColB];
        __syncthreads();

        A += BK;
        B += BK*N;

        for(uint dotIdx = 0; dotIdx < BK; ++dotIdx){
            float tmpB = Bs[dotIdx*BN + threadCol];
            for(uint resIdx = 0; resIdx < TM; ++resIdx){
                threadResults[resIdx] +=
                    As[(threadRow*TM+resIdx)*BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    for(uint resIdx = 0; resIdx < TM; ++resIdx){
        C[(threadRow*TM+resIdx)*N + threadCol] = alpha *threadResults[resIdx] +beta *C[(threadRow*TM+resIdx)*N + threadCol];
    }


    
}

