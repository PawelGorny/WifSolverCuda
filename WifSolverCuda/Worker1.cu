#include "Worker.cuh"

__device__ __constant__ uint64_t _stride[5];
__device__ __shared__ uint32_t _blockResults[4096];
__device__ __shared__ bool _blockResultFlag[1];

__global__ void kernelUncompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks, const uint32_t checksum) {
    uint64_t _start[5];
    beu32 d_hash[8];

    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_start, _stride, tIx);
    _add(_start, buffRangeStart);
    for (uint64_t i = 0, resultIx = tIx; i < threadNumberOfChecks; i++, resultIx++) {
        if (_checksumDoubleSha256CheckUncompressed(checksum, d_hash, _start)) {
            buffResult[resultIx] = true;
            buffCollectorWork[0] = true;
        }
        _add(_start, _stride);        
    }
}
__global__ void kernelCompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks, const uint32_t checksum) {
    uint64_t _start[5];
    beu32 d_hash[8];

    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_start, _stride, tIx);
    _add(_start, buffRangeStart);
    for (uint64_t i = 0, resultIx = tIx; i < threadNumberOfChecks; i++, resultIx++) {
        if (((_start[0] & 0xff00000000) >> 32) != 0x01) {
            _add(_start, _stride);
            continue;
        }
        if (_checksumDoubleSha256CheckCompressed(checksum, d_hash, _start)) {
            buffResult[resultIx] = true;
            buffCollectorWork[0] = true;
        }        
        _add(_start, _stride);
    }    
}
__global__ void kernelUncompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks) {
	uint64_t _start[5];
    beu32 d_hash[8];

    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_start, _stride, tIx);
    _add(_start, buffRangeStart);
	for (uint64_t i = 0, resultIx = tIx ; i < threadNumberOfChecks; i++, resultIx++) {
        if (_checksumDoubleSha256CheckUncompressed(_start[0] & 0xffffffff, d_hash, _start)) {
            buffResult[resultIx] = true;
            buffCollectorWork[0] = true;
        }        
		_add(_start, _stride);
	}
}
__global__ void kernelCompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* const __restrict__  buffRangeStart, const int threadNumberOfChecks) {
    uint64_t _start[5];
    beu32 d_hash[8];

    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_start, _stride, tIx);
    _add(_start, buffRangeStart);
    for (uint64_t i = 0, resultIx = tIx; i < threadNumberOfChecks; i++, resultIx++) {
        if (((_start[0] & 0xff00000000) >> 32) != 0x01) {
            _add(_start, _stride);
            continue;
        }
        if (_checksumDoubleSha256CheckCompressed(_start[0] & 0xffffffff, d_hash, _start)) {
            buffResult[resultIx] = true;
            buffCollectorWork[0] = true;
        }        
        _add(_start, _stride);
    }
}
__global__ void resultCollector(bool* buffResult, uint64_t* buffCombinedResult, const uint64_t threadsInBlockNumberOfChecks) {
    if (buffCombinedResult[blockIdx.x] == 0xffffffffffff) {
        return;
    }
    uint64_t starterI = 0, starter = blockIdx.x * blockDim.x * threadsInBlockNumberOfChecks;
    if (buffCombinedResult[blockIdx.x] != 0) {
        starterI = buffCombinedResult[blockIdx.x] - starter + 1;
        starter = buffCombinedResult[blockIdx.x] + 1;
    }
    for (uint64_t i = starterI, resultIx = starter; i < threadsInBlockNumberOfChecks; i++, resultIx++) {
        if (buffResult[resultIx]) {
            buffCombinedResult[blockIdx.x] = resultIx;
            buffResult[resultIx] = false;
            return;
        }
    }
    buffCombinedResult[blockIdx.x] = 0xffffffffffff;
}

__global__ void kernelUncompressed(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks, const uint32_t checksum) {
    uint64_t _start[5];
    beu32 d_hash[8];

    int64_t resIx = threadIdx.x;
    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_start, _stride, tIx);
    _add(_start, buffRangeStart);
    bool wasResult = false;
    initShared();
    for (uint64_t i = 0, resultIx = tIx; i < threadNumberOfChecks; i++, resultIx++) {
        if (_checksumDoubleSha256CheckUncompressed(checksum, d_hash, _start)) {
            _blockResults[resIx] = resultIx;
            if (!wasResult) {
                _blockResultFlag[0] = true;
            }
            wasResult = true;
            resIx += blockDim.x;
        }
        _add(_start, _stride);
    }
    summaryShared(gpuIx, unifiedResult, isResultFlag);
}
__global__ void kernelUncompressed(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks) {
    uint64_t _start[5];
    beu32 d_hash[8];

    int64_t resIx = threadIdx.x;
    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_start, _stride, tIx);
    _add(_start, buffRangeStart);
    bool wasResult = false;
    initShared();
    for (uint64_t i = 0, resultIx = tIx; i < threadNumberOfChecks; i++, resultIx++) {
        if (_checksumDoubleSha256CheckUncompressed(_start[0] & 0xffffffff, d_hash, _start)) {
            _blockResults[resIx] = resultIx;
            if (!wasResult) {
                _blockResultFlag[0] = true;
            }
            wasResult = true;
            resIx += blockDim.x;
        }
        _add(_start, _stride);
    }
    summaryShared(gpuIx, unifiedResult, isResultFlag);
}
__global__ void kernelCompressed(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks) {
    uint64_t _start[5];
    beu32 d_hash[8];
    int64_t resIx = threadIdx.x;
    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_start, _stride, tIx);
    _add(_start, buffRangeStart);
    bool wasResult = false;
    initShared();
    for (uint64_t i = 0, resultIx = tIx; i < threadNumberOfChecks; i++, resultIx++) {
        if (((_start[0] & 0xff00000000) >> 32) != 0x01) {
            _add(_start, _stride);
            continue;
        }
        if (_checksumDoubleSha256CheckCompressed(_start[0] & 0xffffffff, d_hash, _start)) {
            _blockResults[resIx] = resultIx;
            if (!wasResult) {
                _blockResultFlag[0] = true;
                wasResult = true;
            }            
            resIx += blockDim.x;            
        }
        _add(_start, _stride);
    }
    summaryShared(gpuIx, unifiedResult, isResultFlag);
}
__global__ void kernelCompressed(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks, const uint32_t checksum) {
    uint64_t _start[5];
    beu32 d_hash[8];

    int64_t resIx = threadIdx.x;
    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_start, _stride, tIx);
    _add(_start, buffRangeStart);
    bool wasResult = false;
    initShared();
    for (uint64_t i = 0, resultIx = tIx; i < threadNumberOfChecks; i++, resultIx++) {
        if (((_start[0] & 0xff00000000) >> 32) != 0x01) {
            _add(_start, _stride);
            continue;
        }
        if (_checksumDoubleSha256CheckCompressed(checksum, d_hash, _start)) {
            _blockResults[resIx] = resultIx;
            if (!wasResult) {
                _blockResultFlag[0] = true;
                wasResult = true;
            }
            resIx += blockDim.x;
        }
        _add(_start, _stride);
    }
    summaryShared(gpuIx, unifiedResult, isResultFlag);
}

__device__ __inline__ void initShared() {
    for (int i = threadIdx.x; i < blockDim.x * 4;) {
        _blockResults[i] = UINT32_MAX;
        i += blockDim.x;
    }
    if (threadIdx.x == 0) {
        _blockResultFlag[0] = false;  
    }
    __syncthreads();
}
__device__ __inline__ void summaryShared(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag) {
    __syncthreads();
    if (threadIdx.x == 0 && _blockResultFlag[0]) {
        isResultFlag[gpuIx] = true;
        for (int i = 0, rIx = (blockIdx.x + 4*gpuIx*gridDim.x*blockDim.x); i < blockDim.x * 4; i++) {
            if (_blockResults[i] != UINT32_MAX) {
                unifiedResult[rIx] = _blockResults[i];
                rIx += gridDim.x;
            }
        }
    }
}

__device__  __inline__ bool _checksumDoubleSha256CheckCompressed(unsigned int checksum, beu32* d_hash, uint64_t* _start) {
    sha256Kernel(d_hash,
        _start[4] >> 16,
        (_start[4] & 0x0000ffff) << 16 | _start[3] >> 48,
        (_start[3] & 0xffffffffffff) >> 16,
        (_start[3] & 0x0000ffff) << 16 | _start[2] >> 48,
        (_start[2] & 0xffffffffffff) >> 16,
        (_start[2] & 0x0000ffff) << 16 | _start[1] >> 48,
        (_start[1] & 0xffffffffffff) >> 16,
        (_start[1] & 0x0000ffff) << 16 | _start[0] >> 48,
        ((_start[0] & 0xffffffffffff) >> 16) & 0xffff0000 | 0x8000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x110);

    return _checksumDoubleSha256(checksum, d_hash);
}

__device__  __inline__ bool _checksumDoubleSha256CheckUncompressed(unsigned int checksum, beu32* d_hash, uint64_t* _start) {
    sha256Kernel(d_hash,
        _start[4] >> 8,
        (_start[4] & 0x000000ff) << 24 | _start[3] >> 40,
        (_start[3] & 0xffffffffff) >> 8,
        (_start[3] & 0xff) << 24 | _start[2] >> 40,
        (_start[2] & 0xffffffffff) >> 8,
        (_start[2] & 0xff) << 24 | _start[1] >> 40,
        (_start[1] & 0xffffffffff) >> 8,
        (_start[1] & 0xff) << 24 | _start[0] >> 40,
        ((_start[0] & 0xffffffffff) >> 8) & 0xff000000 | 0x800000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x108);

    return _checksumDoubleSha256(checksum, d_hash);
}

__device__  __inline__ bool _checksumDoubleSha256(unsigned int checksum, beu32* d_hash) {
    sha256Kernel(d_hash, d_hash[0], d_hash[1], d_hash[2], d_hash[3], d_hash[4], d_hash[5],
        d_hash[6], d_hash[7], 0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x100);
    return (checksum == d_hash[0]);
}

__device__  __inline__ void sha256Kernel(beu32* const hash, C16(COMMA, EMPTY)) {
#undef  H
#define H(i,alpha,magic)  beu32 hout##i;

    H8(EMPTY, EMPTY);

#undef  C
#define C(i)              c##i

#undef  H
#define H(i,alpha,magic)  &hout##i

    sha256_chunk0(C16(COMMA, EMPTY), H8(COMMA, EMPTY));

    //
    // SAVE H'S FOR NOW JUST SO NVCC DOESN'T OPTIMIZE EVERYTHING AWAY
    //
#undef  H
#define H(i,alpha,magic)  hash[i] = hout##i;

    H8(EMPTY, EMPTY);
}

__device__  __inline__ void _add(uint64_t* C, uint64_t* A) {
    __Add1(C, A);
}

__device__  __inline__ void _load(uint64_t* C, uint64_t* A) {
    __Load(C, A);
}


__device__  __inline__ void IMult(uint64_t* r, uint64_t* a, int64_t b) {
    uint64_t t[NBBLOCK];
    // Make b positive
    int64_t msk = b >> 63;
    int64_t nmsk = ~msk;
    b = ((-b) & msk) | (b & ~msk);
    USUBO(t[0], a[0] & nmsk, a[0] & msk);
    USUBC(t[1], a[1] & nmsk, a[1] & msk);
    USUBC(t[2], a[2] & nmsk, a[2] & msk);
    USUBC(t[3], a[3] & nmsk, a[3] & msk);
    USUB(t[4], a[4] & nmsk, a[4] & msk);
    Mult2(r, t, b)
}

cudaError_t loadStride(uint64_t* stride){
    return cudaMemcpyToSymbol(_stride, stride, 5 * sizeof(uint64_t));
}