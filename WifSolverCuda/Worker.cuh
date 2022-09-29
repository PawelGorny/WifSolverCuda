
#include <stdint.h>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lib/hash/sha256.cu"
#include "lib/Math.cuh"


__global__ void kernelUncompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks);
__global__ void kernelCompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks);
__global__ void kernelUncompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks, const uint32_t checksum);
__global__ void kernelCompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks, const uint32_t checksum);
__global__ void resultCollector(bool* buffResult, uint64_t* buffCombinedResult, const uint64_t threadsInBlockNumberOfChecks);

__global__ void kernelCompressed(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks);
__global__ void kernelCompressed(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks, const uint32_t checksum);
__global__ void kernelUncompressed(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks);
__global__ void kernelUncompressed(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks, const uint32_t checksum);

__device__ bool _checksumDoubleSha256CheckUncompressed(unsigned int checksum, beu32* d_hash, uint64_t* _start);
__device__ bool _checksumDoubleSha256CheckCompressed(unsigned int checksum, beu32* d_hash, uint64_t* _start);

__device__ bool _checksumDoubleSha256(unsigned int checksum, beu32* d_hash);

__device__ void _add(uint64_t* C, uint64_t* A);
__device__ void _load(uint64_t* C, uint64_t* A);

__device__ void IMult(uint64_t* r, uint64_t* a, int64_t b); 

__device__ void initShared();
__device__ __inline__ void summaryShared(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag);

cudaError_t loadStride(uint64_t* stride);