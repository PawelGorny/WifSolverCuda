
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <set>

#include "lib/ctpl/ctpl_stl.h"
#include "lib/Int.h"
#include "lib/Math.cuh"
#include "lib/util.h"
#include "lib/Bech32.h"
#include "Worker.cuh"

#include "lib/SECP256k1.h"

using namespace std;

void processCandidate(Int& toTest);
void processCandidateThread(int id, uint64_t bit0, uint64_t bit1, uint64_t bit2, uint64_t bit3, uint64_t bit4);
bool readArgs(int argc, char** argv);
bool readFileAddress(const std::string& file_name);
void showHelp();
bool checkDevice();
void listDevices();
void printConfig();
void printFooter();
void decodeWif();
void printSpeed(double speed);
void saveStatus();
void restoreSettings(string fileStatusRestore);
cudaError_t processingUnit(const uint8_t gpuIx, uint64_t** dev_buffRangeStart, Int buffRangeStart, const uint32_t expectedChecksum, uint32_t* buffResultManaged, bool* buffIsResultManaged);

cudaError_t processCuda();
cudaError_t processCudaUnified();
cudaError_t processCudaUnifiedMulti();
cudaError_t executeKernel(uint32_t* _buffResultManaged, bool* _buffIsResultManaged, uint64_t* const _dev_buffRangeStart, const uint32_t _checksum, const int gpuIx);

bool unifiedMemory = true;
const size_t RANGE_TRANSFER_SIZE = NB64BLOCK * sizeof(uint64_t);

int DEVICE_NR = 0;
int nDevices;
unsigned int BLOCK_THREADS = 0;
unsigned int BLOCK_NUMBER = 0;
unsigned int THREAD_STEPS = 5000;

size_t wifLen = 53;
int dataLen = 37;

bool COMPRESSED = false;
Int STRIDE, RANGE_START, RANGE_END, RANGE_START_TOTAL, RANGE_TOTAL;
double RANGE_TOTAL_DOUBLE;
Int loopStride;
Int counter;

bool IS_TARGET_ADDRESS = false;
bool IS_TARGET_ADDRESS_SINGLE = false;
set<string>addresses;
unsigned int addressesLen = 0;

Int CHECKSUM;
bool IS_CHECKSUM = false;

bool DECODE = false;
string WIF_TO_DECODE;

bool RESULT = false;

uint64_t outputSize;

string fileResultPartial = "result_partial.txt";
string fileResult = "result.txt";
string fileStatus = "fileStatus.txt";
int fileStatusInterval = 60;
string fileStatusRestore;
bool isRestore = false;

bool isRANGE_START_TOTAL = false;
bool showDevices = false;
bool p2sh = false;
bool bech32 = false;

bool IS_VERBOSE = false;

Secp256K1* secp;

ctpl::thread_pool pool(2);

int main(int argc, char** argv)
{    
    printf("WifSolver 0.6.2\n\n");
    printf("Use parameter '-h' for help and list of available parameters\n\n");

    if (argc <=1 || readArgs(argc, argv)) {
        showHelp(); 
        printFooter();
        return 0;
    }
    if (showDevices) {
        listDevices();
        printFooter();
        return 0;
    }
    if (DECODE) {
        decodeWif();
        printFooter();
        return 0;
    }
    if (isRestore) {
        restoreSettings(fileStatusRestore);
    }   

    dataLen = COMPRESSED ? 38 : 37;
    if (!isRANGE_START_TOTAL) {
        RANGE_START_TOTAL.Set(&RANGE_START);
    }
    else {
        printf("RESTORE: Starting point: %s\n", RANGE_START.GetBase16().c_str());
    }
    RANGE_TOTAL.Set(&RANGE_END);
    RANGE_TOTAL.Sub(&RANGE_START_TOTAL);
    RANGE_TOTAL_DOUBLE = RANGE_TOTAL.ToDouble();

    if (!checkDevice()) {
        return -1;
    }
    printConfig();

    secp = new Secp256K1();
    secp->Init();

    auto time = std::chrono::system_clock::now();
    std::time_t s_time = std::chrono::system_clock::to_time_t(time);
    std::cout << "Work started at " << std::ctime(&s_time);

    cudaError_t cudaStatus;
    if (unifiedMemory) {
        if (DEVICE_NR == -1) {
            cudaStatus = processCudaUnifiedMulti();
        }
        else {
            cudaStatus = processCudaUnified();
        }
    }
    else {
        cudaStatus = processCuda();
    }
    
    time = std::chrono::system_clock::now();
    s_time = std::chrono::system_clock::to_time_t(time);
    std::cout << "\nWork finished at " << std::ctime(&s_time);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Device reset failed!");
        return 1;
    }

    printFooter();
    return 0;
}

cudaError_t processCudaUnifiedMulti() {
    cudaError_t cudaStatus;
    
    uint64_t* buffStride = new uint64_t[NB64BLOCK];    
    __Load(buffStride, STRIDE.bits64);
    for (int i = 0; i < nDevices; i++) {        
        cudaSetDevice(i);
        loadStride(buffStride);        
    }
    delete buffStride;
    
    const int COLLECTOR_SIZE_MM_PER_GPU = 4 * BLOCK_NUMBER * BLOCK_THREADS;
    const int COLLECTOR_SIZE_MM = COLLECTOR_SIZE_MM_PER_GPU * nDevices;
    const uint32_t expectedChecksum = IS_CHECKSUM ? CHECKSUM.GetInt32() : 0;
    uint64_t counter = 0;

    uint32_t* buffResultManaged = new uint32_t[COLLECTOR_SIZE_MM];
    cudaStatus = cudaMallocManaged(&buffResultManaged, COLLECTOR_SIZE_MM * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed 'buffResultManaged': %s\n", cudaGetErrorString(cudaStatus));
    }
    for (int i = 0; i < COLLECTOR_SIZE_MM; i++) {
        buffResultManaged[i] = UINT32_MAX;
    }
    bool* buffIsResultManaged = new bool[nDevices];
    cudaStatus = cudaMallocManaged(&buffIsResultManaged, nDevices * sizeof(bool));
    for (int gpuIx = 0; gpuIx < nDevices; gpuIx++) {
        buffIsResultManaged[gpuIx] = false;
    }

    uint64_t* buffRangeStart = new uint64_t[NB64BLOCK];
    uint64_t** dev_buffRangeStart = (uint64_t**)malloc(sizeof(uint64_t*) * nDevices);
    for (int gpuIx = 0; gpuIx < nDevices; gpuIx++) {
        cudaSetDevice(gpuIx);
        cudaStatus = cudaMalloc((void**)&dev_buffRangeStart[gpuIx], NB64BLOCK * sizeof(uint64_t));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "%d:: cudaMallocManaged failed 'buffResultManaged': %s\n", gpuIx, cudaGetErrorString(cudaStatus));
        }
    }

    std::thread* threads = new std::thread[nDevices];

    std::chrono::steady_clock::time_point beginCountHashrate = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point beginCountStatus = std::chrono::steady_clock::now();

    while (!RESULT && RANGE_START.IsLower(&RANGE_END)) {
        Int rangeTestStart = new Int(&RANGE_START);     
        //__Load(buffRangeStart, RANGE_START.bits64);
        for (int gpuIx = 0; gpuIx < nDevices; gpuIx++) {            
            threads[gpuIx] = thread(processingUnit, gpuIx, dev_buffRangeStart, rangeTestStart, expectedChecksum, buffResultManaged, buffIsResultManaged);
        }

        long long tHash = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - beginCountHashrate).count();
        if (tHash >= 5000) {
            printSpeed((double)((double)counter / tHash) / 1000.0);
            counter = 0;
            beginCountHashrate = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - beginCountStatus).count() >= fileStatusInterval) {
                saveStatus();
                while (!pool.isQueueEmpty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
                pool.clear_queue();
                beginCountStatus = std::chrono::steady_clock::now();
            }
        }
        counter += outputSize * nDevices;
        for (uint8_t gpuIx = 0; gpuIx < nDevices; gpuIx++) {
            threads[gpuIx].join();                        
        }

        for (uint8_t gpuIx = 0; gpuIx < nDevices; gpuIx++) {
            //test result, to be moved to separate thread
            if (buffIsResultManaged[gpuIx]) {
                buffIsResultManaged[gpuIx] = false;
                for (int i = COLLECTOR_SIZE_MM_PER_GPU *gpuIx, ix=0; ix < COLLECTOR_SIZE_MM_PER_GPU && !RESULT; i++, ix++) {
                    if (buffResultManaged[i] != UINT32_MAX) {
                        Int toTest = new Int(&rangeTestStart);
                        Int diff = new Int(&STRIDE);
                        diff.Mult(buffResultManaged[i]);
                        toTest.Add(&diff);
                        uint64_t bitsToSet[NB64BLOCK];
#pragma unroll NB64BLOCK
                        for (int b = 0; b < NB64BLOCK; b++) {
                            bitsToSet[b] = toTest.bits64[b];
                        }
                        pool.push(processCandidateThread, bitsToSet[0], bitsToSet[1], bitsToSet[2], bitsToSet[3], bitsToSet[4]);
                        //processCandidate(toTest);
                        buffResultManaged[i] = UINT32_MAX;
                    }
                }
            }//test
            rangeTestStart.Add(&loopStride);
            RANGE_START.Add(&loopStride);
        }
    }
    return cudaStatus;
}

cudaError_t processingUnit(const uint8_t gpuIx, uint64_t** dev_buffRangeStart, Int rangeStart, const uint32_t expectedChecksum, uint32_t* buffResultManaged, bool* buffIsResultManaged) {
    cudaSetDevice(gpuIx);
    Int tempStart = new Int(&rangeStart);
    if (gpuIx > 0) {
        Int m = new Int((uint64_t)gpuIx);
        m.Mult(&loopStride);
        tempStart.Add(&m);        
    }    
    uint64_t* tmpBufferStart = new uint64_t[NBBLOCK];
    __Load(tmpBufferStart, tempStart.bits64);
    cudaError_t cudaStatus = cudaMemcpy(dev_buffRangeStart[gpuIx], tmpBufferStart, RANGE_TRANSFER_SIZE, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "%d:: cudaMemcpy failed 'dev_buffRangeStart': %s\n", gpuIx, cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    cudaStatus = executeKernel(buffResultManaged, buffIsResultManaged, dev_buffRangeStart[gpuIx], expectedChecksum, gpuIx);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "%d:: kernel launch failed: %s\n", gpuIx, cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "%d:: cudaDeviceSynchronize returned error code %d after launching kernel!\n", gpuIx, cudaStatus);
        return cudaStatus;
    }
    return cudaStatus;
}

cudaError_t processCudaUnified() {
    cudaError_t cudaStatus;
    uint64_t* buffRangeStart = new uint64_t[NB64BLOCK];
    uint64_t* dev_buffRangeStart = new uint64_t[NB64BLOCK];
    uint64_t* buffStride = new uint64_t[NB64BLOCK];

    const int COLLECTOR_SIZE_MM = 4 * BLOCK_NUMBER * BLOCK_THREADS;
    const uint32_t expectedChecksum = IS_CHECKSUM ? CHECKSUM.GetInt32() : 0;
    uint64_t counter = 0;

    __Load(buffStride, STRIDE.bits64);
    loadStride(buffStride);
    delete buffStride;

    uint32_t* buffResultManaged = new uint32_t[COLLECTOR_SIZE_MM];
    cudaStatus = cudaMallocManaged(&buffResultManaged, COLLECTOR_SIZE_MM * sizeof(uint32_t));
    for (int i = 0; i < COLLECTOR_SIZE_MM; i++) {
        buffResultManaged[i] = UINT32_MAX;
    }

    cudaStatus = cudaMalloc((void**)&dev_buffRangeStart, NB64BLOCK * sizeof(uint64_t));

    bool* buffIsResultManaged = new bool[1];
    cudaStatus = cudaMallocManaged(&buffIsResultManaged, 1 * sizeof(bool));
    buffIsResultManaged[0] = false;    

    __Load(buffRangeStart, RANGE_START.bits64);
    cudaStatus = cudaMemcpy(dev_buffRangeStart, buffRangeStart, RANGE_TRANSFER_SIZE, cudaMemcpyHostToDevice);

    std::chrono::steady_clock::time_point beginCountHashrate = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point beginCountStatus = std::chrono::steady_clock::now();

    while (!RESULT && RANGE_START.IsLower(&RANGE_END)) {
        //launch work
        cudaStatus = executeKernel(buffResultManaged, buffIsResultManaged, dev_buffRangeStart, expectedChecksum, 0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        //status display while waiting for GPU
        long long tHash = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - beginCountHashrate).count();
        if (tHash >= 5000) {
            printSpeed((double)((double)counter / tHash) / 1000.0);
            counter = 0;
            beginCountHashrate = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - beginCountStatus).count() >= fileStatusInterval) {
                saveStatus();
                while (!pool.isQueueEmpty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
                pool.clear_queue();
                beginCountStatus = std::chrono::steady_clock::now();
            }
        }

        counter += outputSize;
        //prepare the results tests
        Int rangeTestStart = new Int(&RANGE_START);
        //pre-prepare the next launch
        RANGE_START.Add(&loopStride);
        __Load(buffRangeStart, RANGE_START.bits64);

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
            goto Error;
        }

        //prepare the next launch on device
        cudaStatus = cudaMemcpyAsync(dev_buffRangeStart, buffRangeStart, RANGE_TRANSFER_SIZE, cudaMemcpyHostToDevice);

        //verify the last results
        if (buffIsResultManaged[0]) {            
            buffIsResultManaged[0] = false;
            for (int i = 0; i < COLLECTOR_SIZE_MM && !RESULT; i++) {
                if (buffResultManaged[i] != UINT32_MAX) {                    
                    Int toTest = new Int(&rangeTestStart);
                    Int diff = new Int(&STRIDE);
                    diff.Mult(buffResultManaged[i]);
                    toTest.Add(&diff);
                    uint64_t bitsToSet[NB64BLOCK];
#pragma unroll NB64BLOCK
                    for (int b = 0; b < NB64BLOCK; b++) {
                        bitsToSet[b] = toTest.bits64[b];
                    }
                    pool.push(processCandidateThread, bitsToSet[0], bitsToSet[1], bitsToSet[2], bitsToSet[3], bitsToSet[4]);                    
                    //processCandidate(toTest);
                    buffResultManaged[i] = UINT32_MAX;                                        
                }                
            }            
        }//test
    }//while loop

Error:
    cudaFree(dev_buffRangeStart);
    cudaFree(buffResultManaged);
    cudaFree(buffIsResultManaged);
    return cudaStatus;
}

cudaError_t executeKernel(uint32_t* _buffResultManaged, bool* _buffIsResultManaged, uint64_t* const _dev_buffRangeStart, const uint32_t _checksum, const int gpuIx) {
    if (COMPRESSED) {
        if (IS_CHECKSUM) {
            kernelCompressed << <BLOCK_NUMBER, BLOCK_THREADS >> > (gpuIx, _buffResultManaged, _buffIsResultManaged, _dev_buffRangeStart, THREAD_STEPS, _checksum);
        }
        else {
            kernelCompressed << <BLOCK_NUMBER, BLOCK_THREADS >> > (gpuIx, _buffResultManaged, _buffIsResultManaged, _dev_buffRangeStart, THREAD_STEPS);
        }
    }
    else {
        if (IS_CHECKSUM) {
            kernelUncompressed << <BLOCK_NUMBER, BLOCK_THREADS >> > (gpuIx, _buffResultManaged, _buffIsResultManaged, _dev_buffRangeStart, THREAD_STEPS, _checksum);
        }
        else {
            kernelUncompressed << <BLOCK_NUMBER, BLOCK_THREADS >> > (gpuIx, _buffResultManaged, _buffIsResultManaged, _dev_buffRangeStart, THREAD_STEPS);
        }
    }
    return cudaGetLastError();
}

cudaError_t processCuda() {
    cudaError_t cudaStatus;
    uint64_t* buffRangeStart = new uint64_t[NB64BLOCK];
    uint64_t* dev_buffRangeStart = new uint64_t[NB64BLOCK];
    uint64_t* buffStride = new uint64_t[NB64BLOCK];
    
    int COLLECTOR_SIZE = BLOCK_NUMBER;

    __Load(buffStride, STRIDE.bits64);
    loadStride(buffStride);
    delete buffStride;


    bool* buffDeviceResult = new bool[outputSize];
    bool* dev_buffDeviceResult = new bool[outputSize];
    for (int i = 0; i < outputSize; i++) {
        buffDeviceResult[i] = false;
    }
    cudaStatus = cudaMalloc((void**)&dev_buffDeviceResult, outputSize * sizeof(bool));
    cudaStatus = cudaMemcpy(dev_buffDeviceResult, buffDeviceResult, outputSize * sizeof(bool), cudaMemcpyHostToDevice);       
        
    delete buffDeviceResult;

    uint64_t* buffResult = new uint64_t[COLLECTOR_SIZE];
    uint64_t* dev_buffResult = new uint64_t[COLLECTOR_SIZE];
    cudaStatus = cudaMalloc((void**)&dev_buffResult, COLLECTOR_SIZE * sizeof(uint64_t));
    cudaStatus = cudaMemcpy(dev_buffResult, buffResult, COLLECTOR_SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);

    bool* buffCollectorWork = new bool[1];
    buffCollectorWork[0] = false;
    bool* dev_buffCollectorWork = new bool[1];
    cudaStatus = cudaMalloc((void**)&dev_buffCollectorWork, 1 * sizeof(bool));
    cudaStatus = cudaMemcpy(dev_buffCollectorWork, buffCollectorWork, 1 * sizeof(bool), cudaMemcpyHostToDevice);

    cudaStatus = cudaMalloc((void**)&dev_buffRangeStart, NB64BLOCK * sizeof(uint64_t));

    const uint32_t expectedChecksum = IS_CHECKSUM ? CHECKSUM.GetInt32() : 0;

    uint64_t counter = 0;
    bool anyResult = false;

    size_t RANGE_TRANSFER_SIZE = NB64BLOCK * sizeof(uint64_t);
    size_t COLLECTOR_TRANSFER_SIZE = COLLECTOR_SIZE * sizeof(uint64_t);

    std::chrono::steady_clock::time_point beginCountHashrate = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point beginCountStatus = std::chrono::steady_clock::now();

    while (!RESULT && RANGE_START.IsLower(&RANGE_END)) {
        //prepare launch
        __Load(buffRangeStart, RANGE_START.bits64);
        cudaStatus = cudaMemcpy(dev_buffRangeStart, buffRangeStart, RANGE_TRANSFER_SIZE, cudaMemcpyHostToDevice);
        //launch work
        if (COMPRESSED) {
            if (IS_CHECKSUM) {
                kernelCompressed << <BLOCK_NUMBER, BLOCK_THREADS >> > (dev_buffDeviceResult, dev_buffCollectorWork, dev_buffRangeStart, THREAD_STEPS, expectedChecksum);
            }else{
                kernelCompressed << <BLOCK_NUMBER, BLOCK_THREADS >> > (dev_buffDeviceResult, dev_buffCollectorWork, dev_buffRangeStart, THREAD_STEPS);
            }            
        }
        else {            
            if (IS_CHECKSUM) {
                kernelUncompressed << <BLOCK_NUMBER, BLOCK_THREADS >> > (dev_buffDeviceResult, dev_buffCollectorWork, dev_buffRangeStart, THREAD_STEPS, expectedChecksum);
            }else{
                kernelUncompressed << <BLOCK_NUMBER, BLOCK_THREADS >> > (dev_buffDeviceResult, dev_buffCollectorWork, dev_buffRangeStart, THREAD_STEPS);
            }
            
        }        
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
            goto Error;
        }

        //if (useCollector) {
            //summarize results            
            cudaStatus = cudaMemcpy(buffCollectorWork, dev_buffCollectorWork, sizeof(bool), cudaMemcpyDeviceToHost);                      
            if (buffCollectorWork[0]) {
                anyResult = true;
                buffCollectorWork[0] = false;                
                cudaStatus = cudaMemcpyAsync(dev_buffCollectorWork, buffCollectorWork, sizeof(bool), cudaMemcpyHostToDevice);            
                for (int i = 0; i < COLLECTOR_SIZE; i++) {
                    buffResult[i] = 0;
                }
                cudaStatus = cudaMemcpy(dev_buffResult, buffResult, COLLECTOR_TRANSFER_SIZE, cudaMemcpyHostToDevice);
                while (anyResult && !RESULT) {
                    resultCollector << <BLOCK_NUMBER, 1 >> > (dev_buffDeviceResult, dev_buffResult, THREAD_STEPS * BLOCK_THREADS);
                    cudaStatus = cudaGetLastError();
                    if (cudaStatus != cudaSuccess) {
                        fprintf(stderr, "kernel 'resultCollector' launch failed: %s\n", cudaGetErrorString(cudaStatus));
                        goto Error;
                    }
                    cudaStatus = cudaDeviceSynchronize();
                    if (cudaStatus != cudaSuccess) {
                        fprintf(stderr, "cudaDeviceSynchronize 'resultCollector' returned error code %d after launching kernel!\n", cudaStatus);
                        goto Error;
                    }
                    cudaStatus = cudaMemcpy(buffResult, dev_buffResult, COLLECTOR_TRANSFER_SIZE, cudaMemcpyDeviceToHost);
                    if (cudaStatus != cudaSuccess) {
                        fprintf(stderr, "cudaMemcpy failed!");
                        goto Error;
                    }
                    anyResult = false;

                    for (int i = 0; i < COLLECTOR_SIZE; i++) {
                        if (buffResult[i] != 0xffffffffffff) {
                            Int toTest = new Int(&RANGE_START);
                            Int diff = new Int(&STRIDE);
                            diff.Mult(buffResult[i]);
                            toTest.Add(&diff);
                            processCandidate(toTest);
                            anyResult = true;
                        }
                    }
                }//while
            }//anyResult to test
        //}
        /*else {
            //pure output, for debug 
            cudaStatus = cudaMemcpy(buffDeviceResult, dev_buffDeviceResult, outputSize * sizeof(bool), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }
            for (int i = 0; i < outputSize; i++) {
                if (buffDeviceResult[i]) {
                    Int toTest = new Int(&RANGE_START);
                    Int diff = new Int(&STRIDE);
                    diff.Mult(i);
                    toTest.Add(&diff);
                    processCandidate(toTest);
                }
            }
        } */      
        RANGE_START.Add(&loopStride);
        counter += outputSize;
        int64_t tHash = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - beginCountHashrate).count();
        //int64_t tStatus = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - beginCountStatus).count();
        if (tHash > 5) {
            double speed = (double)((double)counter / tHash) / 1000000.0;
            printSpeed(speed);            
            counter = 0;
            beginCountHashrate = std::chrono::steady_clock::now();
        }
        if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - beginCountStatus).count() > fileStatusInterval) {
            saveStatus();
            beginCountStatus = std::chrono::steady_clock::now();
        }
    }//while

Error:
    cudaFree(dev_buffResult);
    cudaFree(dev_buffDeviceResult);
    cudaFree(dev_buffRangeStart);
    cudaFree(dev_buffCollectorWork);
    return cudaStatus;
}

void restoreSettings(string fileStatusRestore) {
    const int lineLength = 128;
    char line[lineLength];
    FILE* stat = fopen(fileStatusRestore.c_str(), "r");    
    if (stat == NULL) {
        return;
    }
    while (fgets(line, 128, stat)) {
        string s = string(line);
        std::string prefix("-rangeStart=");
        if (s.rfind(prefix, 0) == 0) {
            RANGE_START.SetBase16((char*)s.substr(prefix.size(), s.size() - prefix.size() - 1).c_str());
            continue;
        }
        prefix = string("-rangeStartInit=");
        if (s.rfind(prefix, 0) == 0) {
            RANGE_START_TOTAL.SetBase16((char*)s.substr(prefix.size(), s.size() - prefix.size() - 1).c_str());
            isRANGE_START_TOTAL = true;
            continue;
        }
        prefix = string("-rangeEnd=");
        if (s.rfind(prefix, 0) == 0) {
            RANGE_END.SetBase16((char*)s.substr(prefix.size(), s.size() - prefix.size() - 1).c_str());
            continue;
        }
        prefix = string("-stride=");
        if (s.rfind(prefix, 0) == 0) {
            STRIDE.SetBase16((char*)s.substr(prefix.size(), s.size() - prefix.size() - 1).c_str());
            continue;
        }
        prefix = string("-a=");
        if (s.rfind(prefix, 0) == 0) {
            addresses.insert(s.substr(prefix.size(), s.size() - prefix.size() - 1));
            IS_TARGET_ADDRESS_SINGLE = true;
            IS_TARGET_ADDRESS = true;
            continue;
        }
        prefix = string("-b=");
        if (s.rfind(prefix, 0) == 0) {
            BLOCK_NUMBER = strtol(s.substr(prefix.size(), s.size() - prefix.size() - 1).c_str(), NULL, 10);
            continue;
        }
        prefix = string("-t=");
        if (s.rfind(prefix, 0) == 0) {
            BLOCK_THREADS = strtol(s.substr(prefix.size(), s.size() - prefix.size() - 1).c_str(), NULL, 10);
            continue;
        }
        prefix = string("-s=");
        if (s.rfind(prefix, 0) == 0) {
            THREAD_STEPS = strtol(s.substr(prefix.size(), s.size() - prefix.size() - 1).c_str(), NULL, 10);
            continue;
        }
        prefix = string("-checksum=");
        if (s.rfind(prefix, 0) == 0) {
            string chxsum = s.substr(prefix.size(), s.size() - prefix.size() - 1);
            if (!chxsum.empty()) {
                CHECKSUM.SetBase16((char*)s.substr(prefix.size(), s.size() - prefix.size() - 1).c_str());
                IS_CHECKSUM = true;
            }
            else {
                IS_CHECKSUM = false;
            }
            continue;
        }
        prefix = string("-c");
        if (s.rfind(prefix) == 0) {
            COMPRESSED = true;
            continue;
        }
        prefix = string("-u");
        if (s.rfind(prefix) == 0) {
            COMPRESSED = false;
            continue;
        }
    }
    fclose(stat);
}

void saveStatus() {
    FILE* stat = fopen(fileStatus.c_str(), "w");    
    auto time = std::chrono::system_clock::now();
    std::time_t s_time = std::chrono::system_clock::to_time_t(time);
    fprintf(stat, "%s\n", std::ctime(&s_time));    
    char wif[53];
    unsigned char* buff = new unsigned char[dataLen];
    for (int i = 0, d = dataLen - 1; i < dataLen; i++, d--) {
        buff[i] = RANGE_START.GetByte(d);
    }
    if (b58encode(wif, &wifLen, buff, dataLen)) {
        fprintf(stat, "%s\n", wif);
    }
    fprintf(stat, "-rangeStartInit=%s\n", RANGE_START_TOTAL.GetBase16().c_str());
    fprintf(stat, "-rangeStart=%s\n", RANGE_START.GetBase16().c_str());
    fprintf(stat, "-rangeEnd=%s\n", RANGE_END.GetBase16().c_str());
    fprintf(stat, "-stride=%s\n", STRIDE.GetBase16().c_str());    
    if (IS_TARGET_ADDRESS_SINGLE) {
        fprintf(stat, "-a=%s\n", (* (addresses.begin())).c_str() );
    }
    if (IS_CHECKSUM) {
        fprintf(stat, "-checksum=%s\n", CHECKSUM.GetBase16().c_str());
    }else {
        fprintf(stat, "-checksum=\n");
    }
    if (COMPRESSED) {
        fprintf(stat, "-c\n");
    }else {
        fprintf(stat, "-u\n");
    }
    fprintf(stat, "-b=%d\n", BLOCK_NUMBER);
    fprintf(stat, "-t=%d\n", BLOCK_THREADS);
    fprintf(stat, "-s=%d\n", THREAD_STEPS);
    fclose(stat);
}

void printSpeed(double speed) {
    std::string speedStr;
    if (speed < 0.01) {
        speedStr = "< 0.01 MKey/s";
    }
    else {
        if (speed < 1000) {
            speedStr = formatDouble("%.3f", speed) + " MKey/s";
        }
        else {
            speed /= 1000;
            if (speed < 1000) {
                speedStr = formatDouble("%.3f", speed) + " GKey/s";
            }
            else {
                speed /= 1000;
                speedStr = formatDouble("%.3f", speed) + " TKey/s";
            }
        }
    }

    Int processedCount= new Int(&RANGE_START);
    processedCount.Sub(&RANGE_START_TOTAL);
    double _count = processedCount.ToDouble(); 
    _count = _count / RANGE_TOTAL_DOUBLE;
    _count *= 100;
    if (IS_VERBOSE) {
        char wif[53];
        unsigned char* buff = new unsigned char[dataLen];
        for (int i = 0, d = dataLen - 1; i < dataLen; i++, d--) {
            buff[i] = RANGE_START.GetByte(d);
        }
        b58encode(wif, &wifLen, buff, dataLen);
        printf("\r %s,  progress: %.3f%%, WIF: %s    ", speedStr.c_str(), _count, wif);
    }
    else {
        printf("\r %s,  progress: %.3f%%     ", speedStr.c_str(), _count);
    }
    fflush(stdout);
}

void processCandidate(Int &toTest) {     
    FILE* keys;
    char rmdhash[21], address[128], wif[53];

    unsigned char* buff = new unsigned char[dataLen];
    for (int i = 0, d=dataLen-1; i < dataLen; i++, d--) {
        buff[i] = toTest.GetByte(d);
    }       
    toTest.SetBase16((char*)toTest.GetBase16().substr(2, 64).c_str());        
    Point publickey = secp->ComputePublicKey(&toTest);
    if (bech32){
        char output[128];
        uint8_t h160[20];
        secp->GetHash160(BECH32, true, publickey, h160);
        segwit_addr_encode(output, "bc", 0, h160, 20);
        string addressBech32 = std::string(output);
        strcpy(address, addressBech32.c_str());
    }
    else {
        if (p2sh) {
            secp->GetHash160(P2SH, true, publickey, (unsigned char*)rmdhash);
        }
        else {
            secp->GetHash160(P2PKH, COMPRESSED, publickey, (unsigned char*)rmdhash);
        }
        addressToBase58(rmdhash, address, p2sh);
    }   
    if (IS_TARGET_ADDRESS) {
        if (addresses.find(address) != addresses.end()) {
            RESULT = true;            
            printf("\n");
            printf("found: %s\n", address);
            printf("key  : %s\n", toTest.GetBase16().c_str());
            if (b58encode(wif, &wifLen, buff, dataLen)) {
                printf("WIF  : %s\n", wif);
            }
            keys = fopen(fileResult.c_str(), "a+");
            fprintf(keys, "%s\n", address);
            fprintf(keys, "%s\n", wif);
            fprintf(keys, "%s\n\n", toTest.GetBase16().c_str());            
            fclose(keys);
            return;
        }
    }
    else {
        printf("\n");
        printf("found: %s\n", address);
        printf("key  : %s\n", toTest.GetBase16().c_str());
        if (b58encode(wif, &wifLen, buff, dataLen)) {
            printf("WIF  : %s\n", wif);
        }
        keys = fopen(fileResultPartial.c_str(), "a+");
        fprintf(keys, "%s\n", address);
        fprintf(keys, "%s\n", wif);
        fprintf(keys, "%s\n\n", toTest.GetBase16().c_str());        
        fclose(keys);
    }
}

void processCandidateThread(int id, uint64_t bit0, uint64_t bit1, uint64_t bit2, uint64_t bit3, uint64_t bit4) {
    Int* toTest;
    toTest = new Int();
    toTest->bits64[4] = bit4;
    toTest->bits64[3] = bit3;   toTest->bits64[2] = bit2;
    toTest->bits64[1] = bit1;   toTest->bits64[0] = bit0;
    processCandidate(*toTest);
    delete toTest;
}

void printConfig() {
    printf("Range start: %s\n", RANGE_START_TOTAL.GetBase16().c_str());
    printf("Range end  : %s\n", RANGE_END.GetBase16().c_str());
    printf("Stride     : %s\n", STRIDE.GetBase16().c_str());
    if (IS_CHECKSUM) {
        printf("Checksum   : %s\n", CHECKSUM.GetBase16().c_str());
    }
    if (IS_TARGET_ADDRESS_SINGLE) {
        printf("Target     : %s\n", (*(addresses.begin())).c_str());
    }
    else {
        printf("Targets    : %d\n", addressesLen);
    }
    if (COMPRESSED) {
        printf("Target COMPRESSED\n");
    }    else    {
        printf("Target UNCOMPRESSED\n");
    }       
    printf( "\n");
    printf( "number of blocks: %d\n", BLOCK_NUMBER);
    printf( "number of threads: %d\n", BLOCK_THREADS);
    printf( "number of checks per thread: %d\n", THREAD_STEPS);
    printf( "\n");
}

void printFooter() {
    printf("------------------------\n");
    printf("source: https://github.com/PawelGorny/WifSolverCuda\n\n");
}

bool checkDevice() {
    if (DEVICE_NR == -1) {
        cudaGetDeviceCount(&nDevices);
        //nDevices = 1;
        for (int i = 0; i < nDevices; i++) {
            cudaError_t cudaStatus = cudaSetDevice(i);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "device %d failed!", i);
                return false;
            }
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, i);
            printf("Device Number: %d\n", i);
            printf("  %s\n", props.name);
            if (BLOCK_NUMBER == 0) {
                BLOCK_NUMBER = props.multiProcessorCount * 4;
            }
            if (BLOCK_THREADS == 0) {
                BLOCK_THREADS = props.maxThreadsPerBlock / 8 * 3;
            }
        }
        outputSize = BLOCK_NUMBER * BLOCK_THREADS * THREAD_STEPS;
        loopStride = new Int(&STRIDE);
        loopStride.Mult(outputSize);
        return true;
    }
    else {
        cudaError_t cudaStatus = cudaSetDevice(DEVICE_NR);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "device %d failed!", DEVICE_NR);
            return false;
        }
        else {
            cudaDeviceProp props;
            cudaStatus = cudaGetDeviceProperties(&props, DEVICE_NR);
            printf("Using GPU nr %d:\n", DEVICE_NR);
            if (props.canMapHostMemory == 0) {
                printf("unified memory not supported\n");
                unifiedMemory = 0;
            }
            printf("%s (%2d procs)\n", props.name, props.multiProcessorCount);
            printf("maxThreadsPerBlock: %2d\n\n", props.maxThreadsPerBlock);
            if (BLOCK_NUMBER == 0) {
                BLOCK_NUMBER = props.multiProcessorCount * 8;
            }
            if (BLOCK_THREADS == 0) {
                BLOCK_THREADS = (props.maxThreadsPerBlock / 8) * 5;
            }
            outputSize = BLOCK_NUMBER * BLOCK_THREADS * THREAD_STEPS;
            loopStride = new Int(&STRIDE);
            loopStride.Mult(outputSize);
        }
        return true;
    }
}

void showHelp()  {    
    printf("WifSolverCuda [-d deviceId] [-b NbBlocks] [-t NbThreads] [-s NbThreadChecks]\n");
    printf("    [-fresultp reportFile] [-fresult resultFile] [-fstatus statusFile] [-a targetAddress]\n");
    printf("    -stride hexKeyStride -rangeStart hexKeyStart [-rangeEnd hexKeyEnd] [-checksum hexChecksum] \n");
    printf("    -wifStart wifKeyStart [-wifEnd wifKeyEnd]\n");
    printf("    [-decode wifToDecode] \n");
    printf("    [-restore statusFile] \n");
    printf("    [-listDevices] \n");
    printf("    [-v] \n");
    printf("    [-h] \n\n");
    printf("-rangeStart hexKeyStart: decoded initial key with compression flag and checksum \n");
    printf("-rangeEnd hexKeyEnd:     decoded end key with compression flag and checksum (optional)\n");
    printf("-wifStart wifKeyStart:   initial key (WIF)\n");
    printf("-wifEnd wifKeyEnd:       end key (WIF) (optional)\n");
    printf("-checksum hexChecksum:   decoded checksum, cannot be modified with a stride  \n");
    printf("-stride hexKeyStride:    full stride calculated as 58^(most-right missing char index) \n");
    printf("-a targetAddress:        expected address\n");
    printf("-afile file:             file with target addresses (all the same type, 1.., bc..., 3...)\n");
    printf("-fresult resultFile:     file for final result (default: %s)\n", fileResult.c_str());
    printf("-fresultp reportFile:    file for each WIF with correct checksum (default: %s)\n", fileResultPartial.c_str());
    printf("-fstatus statusFile:     file for periodically saved status (default: %s) \n", fileStatus.c_str());
    printf("-fstatusIntv seconds:    period between status file updates (default %d sec) \n", fileStatusInterval);
    printf("-d deviceId:             default 0, '-d all' for all available CUDA devices\n");
    printf("-c :                     search for compressed address\n");
    printf("-u :                     search for uncompressed address (default)\n");
    printf("-b NbBlocks:             default processorCount * 8\n");
    printf("-t NbThreads:            default deviceMax/8 * 5\n");
    printf("-s NbThreadChecks:       default %d\n", THREAD_STEPS);
    printf("-decode wifToDecode:     decodes given WIF\n");    
    printf("-restore statusFile:     restore work configuration\n");
    printf("-listDevices:            shows available devices\n");
    printf("-disable-um:             disable unified memory mode\n");
    printf("-v :                     verbose output\n");
    printf("-h :                     shows help\n");
}
 
bool readArgs(int argc, char** argv) {
    int a = 1;
    bool isStride = false;
    bool isStart = false;
    bool isEnd = false;    
    while (a < argc) {
        if (strcmp(argv[a], "-h") == 0) {
            return true;
        }else
        if (strcmp(argv[a], "-restore") == 0) {
            a++;
            fileStatusRestore = string(argv[a]);
            isRestore = true;
            return false;
        }else
        if (strcmp(argv[a], "-decode") == 0) {
            a++;
            WIF_TO_DECODE = string(argv[a]);
            DECODE = true;
            return false;
        }
        else if (strcmp(argv[a], "-listDevices") == 0) {
            showDevices = true;
            return false;
        }
        else if (strcmp(argv[a], "-d") == 0) {
            a++;
            if ("all" == string(argv[a])) {
                DEVICE_NR = -1;
            }
            else {
                DEVICE_NR = strtol(argv[a], NULL, 10);
            }
        }
        else if (strcmp(argv[a], "-c") == 0) {
            COMPRESSED = true;
        }
        else if (strcmp(argv[a], "-u") == 0) {
            COMPRESSED = false;
            if (p2sh) {
                COMPRESSED = true;
            }
        }
        else if (strcmp(argv[a], "-t") == 0) {
            a++;
            BLOCK_THREADS = strtol(argv[a], NULL, 10);
        }
        else if (strcmp(argv[a], "-b") == 0) {
            a++;
            BLOCK_NUMBER = strtol(argv[a], NULL, 10);
        }
        else if (strcmp(argv[a], "-s") == 0) {
            a++;
            THREAD_STEPS = strtol(argv[a], NULL, 10);
        }
        else if (strcmp(argv[a], "-stride") == 0) {
            a++;
            STRIDE.SetBase16((char*)string(argv[a]).c_str());
            isStride = true;
        }
        else if (strcmp(argv[a], "-rangeStart") == 0) {
            a++;
            RANGE_START.SetBase16((char*)string(argv[a]).c_str());
            isStart = true;
        }
        else if (strcmp(argv[a], "-rangeEnd") == 0) {
            a++;
            RANGE_END.SetBase16((char*)string(argv[a]).c_str());
            isEnd = true;
        }
        else if (strcmp(argv[a], "-wifStart") == 0) {
            a++;
            WIF_TO_DECODE = string(argv[a]);
            const char* base58 = WIF_TO_DECODE.c_str();
            size_t base58Length = WIF_TO_DECODE.size();
            size_t keybuflen = base58Length == 52 ? 38 : 37;
            unsigned char* keybuf = new unsigned char[keybuflen];
            b58decode(keybuf, &keybuflen, base58, base58Length);
            stringstream ss;
            for (int i = 0; i < keybuflen; ++i) {
                ss << std::setfill('0') << std::setw(2)<< (std::hex) << (int)keybuf[i];
            }
            RANGE_START.SetBase16((char*)ss.str().c_str());
            isStart = true;
        }
        else if (strcmp(argv[a], "-wifEnd") == 0) {
            a++;
            WIF_TO_DECODE = string(argv[a]);
            const char* base58 = WIF_TO_DECODE.c_str();
            size_t base58Length = WIF_TO_DECODE.size();
            size_t keybuflen = base58Length == 52 ? 38 : 37;
            unsigned char* keybuf = new unsigned char[keybuflen];
            b58decode(keybuf, &keybuflen, base58, base58Length);
            stringstream ss;
            for (int i = 0; i < keybuflen; ++i) {
                ss << std::setfill('0') << std::setw(2) << (std::hex) << (int)keybuf[i];
            }
            RANGE_END.SetBase16((char*)ss.str().c_str());
            isEnd = true;
        }
        else if (strcmp(argv[a], "-fresult") == 0) {
            a++;
            fileResult = string(argv[a]);
        }
        else if (strcmp(argv[a], "-fresultp") == 0) {
            a++;
            fileResultPartial = string(argv[a]);
        }
        else if (strcmp(argv[a], "-fstatus") == 0) {
            a++;
            fileStatus = string(argv[a]);
        }
        else if (strcmp(argv[a], "-fstatusIntv") == 0) {
            a++;
            fileStatusInterval = strtol(argv[a], NULL, 10);
        }
        else if (strcmp(argv[a], "-afile") == 0) {
            a++;            
            if (!readFileAddress(string(argv[a]))) {
                return true;
            }
            IS_TARGET_ADDRESS = true;
            IS_TARGET_ADDRESS_SINGLE = (addressesLen == 1);
        }
        else if (strcmp(argv[a], "-a") == 0) {
            a++;
            IS_TARGET_ADDRESS = true;
            IS_TARGET_ADDRESS_SINGLE = true;
            addressesLen = 1;
            addresses.insert(string(argv[a]));
            if (argv[a][0] == '3') {
                p2sh = true;
                COMPRESSED = true;
            }else
            if (argv[a][0] == 'b') {
                bech32 = true;
                COMPRESSED = true;
            }
        }
        else if (strcmp(argv[a], "-checksum") == 0) {
            a++;
            CHECKSUM.SetBase16((char*)string(argv[a]).c_str());
            IS_CHECKSUM = true;
        }
        else if (strcmp(argv[a], "-disable-um") == 0) {
            unifiedMemory = 0;
            printf("unified memory mode disabled\n");
        }
        else if (strcmp(argv[a], "-v") == 0) {
        IS_VERBOSE = true;
        }
        a++;
    }    

    if (!isStart) {
        if (COMPRESSED) {
            RANGE_START.SetBase16((char*)string("800000000000000000000000000000000000000000000000000000000000000001014671fc3f").c_str());
        }
        else {
            RANGE_START.SetBase16((char*)string("800000000000000000000000000000000000000000000000000000000000000001a85aa87e").c_str());
        }
    }
    if (!isEnd) {
        if (COMPRESSED) {
            RANGE_END.SetBase16((char*)string("80fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd036414101271aa03f").c_str());
        }
        else {
            RANGE_END.SetBase16((char*)string("80fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141e4770308").c_str());
        }
    }
    if (!isStride) {
        if (COMPRESSED) {
            STRIDE.SetBase16((char*)string("10000000000").c_str());
        }
        else {
            STRIDE.SetBase16((char*)string("100000000").c_str());
        }
    }    
    return false;
}

void decodeWif() {
    const char* base58 = WIF_TO_DECODE.c_str();
    size_t base58Length = WIF_TO_DECODE.size();
    size_t keybuflen = base58Length == 52 ? 38 : 37;
    unsigned char * keybuf = new unsigned char[keybuflen];
    b58decode(keybuf, &keybuflen, base58, base58Length);
    printf("WIF: %s\n", WIF_TO_DECODE.c_str());
    printf("Decoded:\n");
    for (int i = 0; i < keybuflen; i++) {
        printf("%.2x", keybuf[i]);
    }
    printf("\n\n");
}

void listDevices() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  %s\n", prop.name);
        if (prop.canMapHostMemory == 0) {
            printf("  unified memory not supported\n");
        }
        printf("  %2d procs\n", prop.multiProcessorCount);
        printf("  maxThreadsPerBlock: %2d\n", prop.maxThreadsPerBlock);
        printf("  version majorminor: %d%d\n\n", prop.major, prop.minor);
    }
}

bool readFileAddress(const std::string& file_name) {
    std::ifstream stream(file_name);
    if (IS_VERBOSE) {
        std::cout << "Opening address file '" << file_name << "'" << std::endl;
    }
    if (stream.fail() || !stream.good())
    {
        std::cout << "Error: Failed to open file '" << file_name << "'" << std::endl;
        return false;
    }
    std::string buffer;
    int nr = 0;
    while (!stream.eof() && stream.good() && stream.peek() != EOF)
    {
        std::getline(stream, buffer);
        addresses.insert(buffer);
        if (0 == nr) {
            if (buffer[0] == '3') {
                p2sh = true;
                COMPRESSED = true;
            }
            else
                if (buffer[0] == 'b') {
                    bech32 = true;
                    COMPRESSED = true;
                }
        }
        nr++;
    }
    stream.close();
    addressesLen = nr;
    return true;
}