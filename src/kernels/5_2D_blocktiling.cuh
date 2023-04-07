#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM*BN)/(TM*TN),1)
    sgemm_2D_blocktiling(int M, int N, int K, float alpha, const float*A, const float *B, float beta, float *C){
        const uint cRow = blockIdx.y;
        const uint cCol = blockIdx.x;

        const uint totalResultsBlocktile = BM * BN;
        // number of threads in a block
        // A thread is responsible for calculating TM*TN elements in the blocktile
        const uint numThreadsBlocktile = totalResultsBlocktile / (TM*TN);

        assert(numThreadsBlocktile == blockDim.x);

        // BN/TN is the number of threads for each column
        const int threadCol = threadIdx.x %(BN/TN);
        const int threadRow = threadIdx.x / (BN/TN);

        // allocate space for the current blocktile in smem
        __shared__ float As[BM*BK];
        __shared__ float Bs[BK*BN];

        // move blocktile to the beginning of A's row and B's column
        A += cRow*BM*K;
        B += cCol* BN;
        C += cRow * BM *N + cCol*BN;

        // get the indices of thread which will load into SMEM
        const uint innerRowA = threadIdx.x /BK;
        const uint innerColA = threadIdx.x % BK;
        // get the number of rows of As that are being loaded in a single step
        // by a single block
        const uint strideA = numThreadsBlocktile/BK;

        const uint innerRowB = threadIdx.x /BN;
        const uint innerColB = threadIdx.x % BN;
        const uint strideB = numThreadsBlocktile/BN;

        // allocate thread-local cache for results in register file
        float threadResults[TM*TN] = {0.0};
        // register cache for As and Bs
        float regM[TM] = {0.0};
        float regN[TN] = {0.0};

        for(uint bkIdx = 0; bkIdx <K; bkIdx +=BK){
            for(uint loadOffset = 0; loadOffset < BM; loadOffset += strideA){
                As[(innerRowA + loadOffset) *BK +innerColA] =
                    A[(innerRowA + loadOffset)*K + innerColA];

            }

            for(uint loadOffset = 0; loadOffset < BK; loadOffset += strideB){
                Bs[(innerRowB + loadOffset)* BN + innerColB] =
                    B[(innerRowB + loadOffset)* N + innerColB];
            }
            __syncthreads();

            A += BK;
            B += BK*N;

            for(uint dotIdx = 0; dotIdx < BK; ++dotIdx){
                for(uint i = 0; i < TM; ++i){
                    regM[i] = As[(threadRow* TM +i)* BK+dotIdx];  
                }

                for(uint i = 0; i < TN; ++i){
                    regN[i] = Bs[dotIdx* BN + threadCol* TN+i];  
                }

                for(uint resIdxM = 0; resIdxM < TM; ++resIdxM){
                    for(uint resIdxN = 0; resIdxN < TN; ++resIdxN){
                        threadResults[resIdxM* TN + resIdxN] +=
                         regM[resIdxM] * regN[resIdxN];
                    }
                }
            }
            __syncthreads();
        }

        for(uint resIdxM = 0; resIdxM < TM; ++resIdxM){
            for(uint resIdxN = 0; resIdxN < TN; ++resIdxN){
                C[(threadRow*TM + resIdxM)*N + threadCol*TN+resIdxN] = 
                    alpha * threadResults[resIdxM * TN + resIdxN] +
                    beta * C[(threadRow*TM + resIdxM)*N + threadCol*TN+resIdxN];

            }
        }


    }