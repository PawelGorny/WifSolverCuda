#include "Worker.cuh"

__global__ void kernelUncompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* buffRangeStart, uint64_t* buffStride, const int threadNumberOfChecks, const uint32_t checksum) {
    uint64_t _stride[5];
    uint64_t _start[5];
    uint64_t _startStride[5];
    beu32 d_hash[8];
    _load(_start, buffRangeStart);
    _load(_stride, buffStride);

    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_startStride, _stride, tIx);
    _add(_start, _startStride);    
    for (uint64_t i = 0, resultIx = tIx; i < threadNumberOfChecks; i++, resultIx++) {
        if (_checksumDoubleSha256CheckUncompressed(checksum, d_hash, _start)) {
            buffResult[resultIx] = true;
            buffCollectorWork[0] = true;
        }
        _add(_start, _stride);        
    }
}
__global__ void kernelCompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* buffRangeStart, uint64_t* buffStride, const int threadNumberOfChecks, const uint32_t checksum) {
    uint64_t _stride[5];
    uint64_t _start[5];
    uint64_t _startStride[5];
    beu32 d_hash[8];
    _load(_start, buffRangeStart);
    _load(_stride, buffStride);

    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_startStride, _stride, tIx);
    _add(_start, _startStride);    
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
__global__ void kernelUncompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* buffRangeStart, uint64_t* buffStride, const int threadNumberOfChecks) {
	uint64_t _stride[5];
	uint64_t _start[5];
    uint64_t _startStride[5];
    beu32 d_hash[8];
    _load(_start, buffRangeStart);
    _load(_stride, buffStride);

    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_startStride, _stride, tIx);
    _add(_start, _startStride);
	for (uint64_t i = 0, resultIx = tIx ; i < threadNumberOfChecks; i++, resultIx++) {
        if (_checksumDoubleSha256CheckUncompressed(_start[0] & 0xffffffff, d_hash, _start)) {
            buffResult[resultIx] = true;
            buffCollectorWork[0] = true;
        }        
		_add(_start, _stride);
	}
}
__global__ void kernelCompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* buffRangeStart, uint64_t* buffStride, const int threadNumberOfChecks) {
    uint64_t _stride[5];
    uint64_t _start[5];
    uint64_t _startStride[5];
    beu32 d_hash[8];
    _load(_start, buffRangeStart);
    _load(_stride, buffStride);

    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_startStride, _stride, tIx);
    _add(_start, _startStride);    
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
    int64_t tIx = blockIdx.x * blockDim.x ;
    buffCombinedResult[blockIdx.x] = 0xffffffffffff;
    for (uint64_t i = 0, resultIx = tIx * threadsInBlockNumberOfChecks; i < threadsInBlockNumberOfChecks; i++, resultIx++) {
        if (buffResult[resultIx]) {
            buffCombinedResult[blockIdx.x] = resultIx;
            buffResult[resultIx] = false;
            return;
        }
    }
}

__device__ bool _checksumDoubleSha256CheckCompressed(unsigned int checksum, beu32* d_hash, uint64_t* _start) {
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

__device__ bool _checksumDoubleSha256CheckUncompressed(unsigned int checksum, beu32* d_hash, uint64_t* _start) {
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

__device__ bool _checksumDoubleSha256(unsigned int checksum, beu32* d_hash) {
    sha256Kernel(d_hash, d_hash[0], d_hash[1], d_hash[2], d_hash[3], d_hash[4], d_hash[5],
        d_hash[6], d_hash[7], 0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x100);
    return (checksum == d_hash[0]);
}

__device__ void sha256Kernel(beu32* const hash, C16(COMMA, EMPTY)) {
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

__device__ void _add(uint64_t* C, uint64_t* A) {
    __Add1(C, A);
}

__device__ void _load(uint64_t* C, uint64_t* A) {
    __Load(C, A);
}


__device__ void IMult(uint64_t* r, uint64_t* a, int64_t b) {
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

