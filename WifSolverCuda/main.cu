
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <chrono>

#include "lib/Int.h"
#include "lib/Math.cuh"
#include "lib/util.h"
#include "Worker.cuh"

#include "lib/SECP256k1.h"


using namespace std;

void processCandidate(Int& toTest);
bool readArgs(int argc, char** argv);
void showHelp();
bool checkDevice();
void printConfig();
void printFooter();

cudaError_t processCuda();


int DEVICE_NR = 0;
unsigned int BLOCK_THREADS = 0;
unsigned int BLOCK_NUMBER = 0;
unsigned int THREAD_STEPS = 3364;

bool COMPRESSED = false;
Int STRIDE, RANGE_START, RANGE_END;
Int loopStride;
Int counter;
string TARGET_ADDRESS = "";

bool RESULT = false;
bool useCollector = false;
uint64_t collectorLimit = 100000000;

uint64_t outputSize;

string fileResultPartial = "result_partial.txt";
string fileResult = "result.txt";
string fileStatus = "fileStatus.txt";

Secp256K1* secp;


int main(int argc, char** argv)
{    
    printf("WifSolver 0.2\n\n");

    if (readArgs(argc, argv)) {
        showHelp(); 
        printFooter();
        return 0;
    }

    if (!checkDevice()) {
        return -1;
    }
    printConfig();

    secp = new Secp256K1();
    secp->Init();

    auto time = std::chrono::system_clock::now();
    std::time_t s_time = std::chrono::system_clock::to_time_t(time);
    std::cout << "Work started at " << std::ctime(&s_time);

    cudaError_t cudaStatus = processCuda();
    
    time = std::chrono::system_clock::now();
    s_time = std::chrono::system_clock::to_time_t(time);
    std::cout << "Work finished at " << std::ctime(&s_time);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Device reset failed!");
        return 1;
    }

    printFooter();
    return 0;
}

cudaError_t processCuda() {
    cudaError_t cudaStatus;
    uint64_t* buffRangeStart = new uint64_t[NB64BLOCK];
    uint64_t* dev_buffRangeStart = new uint64_t[NB64BLOCK];
    uint64_t* buffStride = new uint64_t[NB64BLOCK];
    uint64_t* dev_buffStride = new uint64_t[NB64BLOCK];    
    
    __Load(buffStride, STRIDE.bits64);

    cudaStatus = cudaMalloc((void**)&dev_buffRangeStart, NB64BLOCK * sizeof(uint64_t));
    cudaStatus = cudaMalloc((void**)&dev_buffStride, NB64BLOCK * sizeof(uint64_t));
    cudaStatus = cudaMemcpy(dev_buffStride, buffStride, NB64BLOCK * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    bool* buffDeviceResult = new bool[outputSize];
    bool* dev_buffDeviceResult = new bool[outputSize];
    cudaStatus = cudaMalloc((void**)&dev_buffDeviceResult, outputSize * sizeof(bool));
    cudaStatus = cudaMemcpy(dev_buffDeviceResult, buffDeviceResult, outputSize * sizeof(bool), cudaMemcpyHostToDevice);
    
    int COLLECTOR_SIZE = BLOCK_NUMBER;
    uint64_t* buffResult = new uint64_t[COLLECTOR_SIZE];
    uint64_t* dev_buffResult = new uint64_t[COLLECTOR_SIZE];
    cudaStatus = cudaMalloc((void**)&dev_buffResult, COLLECTOR_SIZE * sizeof(uint64_t));
    cudaStatus = cudaMemcpy(dev_buffResult, buffResult, COLLECTOR_SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint64_t counter = 0;
    int counterSaveFile = 0;
        
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    while (!RESULT && RANGE_START.IsLower(&RANGE_END)) {

        //prepare launch
        __Load(buffRangeStart, RANGE_START.bits64);
        cudaStatus = cudaMemcpy(dev_buffRangeStart, buffRangeStart, NB64BLOCK * sizeof(uint64_t), cudaMemcpyHostToDevice);
        //launch work
        if (COMPRESSED) {
            kernelCompressed << <BLOCK_NUMBER, BLOCK_THREADS >> > (dev_buffDeviceResult, dev_buffRangeStart, dev_buffStride, THREAD_STEPS);
        }
        else {            
            kernelUncompressed << <BLOCK_NUMBER, BLOCK_THREADS >> > (dev_buffDeviceResult, dev_buffRangeStart, dev_buffStride, THREAD_STEPS);
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
        //std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() << "[ms]" << std::endl;

        if (useCollector) {
            //summarize results
            bool anyResult = true;
            while (anyResult) {
                resultCollector << <BLOCK_NUMBER, 1 >> > (dev_buffDeviceResult, dev_buffResult, THREAD_STEPS * BLOCK_THREADS);
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
                cudaStatus = cudaMemcpy(buffResult, dev_buffResult, BLOCK_NUMBER * sizeof(uint64_t), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }
                anyResult = false;
                for (int i = 0; i < COLLECTOR_SIZE; i++) {
                    if (buffResult[i] > 0) {
                        Int toTest = new Int(&RANGE_START);
                        Int diff = new Int(&STRIDE);
                        diff.Mult(buffResult[i]);
                        toTest.Add(&diff);
                        processCandidate(toTest);   
                        anyResult = true;
                    }
                }
            }
        }
        else {
            //pure output
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
        }       
        //std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() << "[ms]" << std::endl;        
        RANGE_START.Add(&loopStride);
        counter += outputSize;
        int32_t t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();
        if ( t > 5000) {
            double speed = (double)((double)counter / (t/1000) ) / 1000000.0;            
            std::string speedStr;
            if (speed < 0.01) {
                speedStr = "< 0.01 MKey/s";
            }
            else {
                speedStr = formatDouble("%.2f", speed) + " MKey/s";
            }
            printf("\r %s", speedStr.c_str());
            fflush(stdout);
            counter = 0;
            begin = std::chrono::steady_clock::now();
            counterSaveFile++;
        }
        if (counterSaveFile == 12) {
            counterSaveFile = 0;
            FILE* stat = fopen(fileStatus.c_str(), "w");
            fprintf(stat, "%s\n", RANGE_START.GetBase16().c_str());           
            auto time = std::chrono::system_clock::now();
            std::time_t s_time = std::chrono::system_clock::to_time_t(time);
            fprintf(stat, "%s\n", std::ctime(&s_time));
            fclose(stat);
        }

    }//while

Error:
    cudaFree(dev_buffResult);
    cudaFree(dev_buffDeviceResult);
    cudaFree(dev_buffRangeStart);
    cudaFree(dev_buffStride);
    return cudaStatus;
}


void processCandidate(Int &toTest) {     
    FILE* keys;
    char rmdhash[21], address[50];    
    toTest.SetBase16((char*)toTest.GetBase16().substr(2, 64).c_str());        
    Point publickey = secp->ComputePublicKey(&toTest);        
    secp->GetHash160(P2PKH, COMPRESSED, publickey, (unsigned char*)rmdhash);
    addressToBase58(rmdhash, address);    
    if (!TARGET_ADDRESS.empty()) {
        if (TARGET_ADDRESS._Equal(address)) {
            RESULT = true;            
            printf("\n");
            printf("found: %s\n", address);
            printf("key  : %s\n", toTest.GetBase16().c_str());
            keys = fopen(fileResult.c_str(), "a+");
            fprintf(keys, "%s\n", address);
            fprintf(keys, "%s\n", toTest.GetBase16().c_str());
            fclose(keys);
            return;
        }
    }
    else {
        printf("\n");
        printf("found: %s\n", address);
        printf("key  : %s\n", toTest.GetBase16().c_str());
        keys = fopen(fileResultPartial.c_str(), "a+");
        fprintf(keys, "%s\n", address);
        fprintf(keys, "%s\n", toTest.GetBase16().c_str());
        fclose(keys);
    }
}



void printConfig() {
    printf("Range start: %s\n", RANGE_START.GetBase16().c_str());
    printf( "Range end  : %s\n", RANGE_END.GetBase16().c_str());
    printf( "Stride     : %s\n", STRIDE.GetBase16().c_str());
    if (!TARGET_ADDRESS.empty()) {
        printf( "Target     : %s\n", TARGET_ADDRESS.c_str());
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
    printf("source: https://github.com/PawelGorny/WifSolverCuda\n");
    printf("If you found this program useful, please donate - bitcoin:34dEiyShGJcnGAg2jWhcoDDRxpennSZxg8\n");
}

bool checkDevice() {
    cudaError_t cudaStatus = cudaSetDevice(DEVICE_NR);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "device %d failed!", DEVICE_NR);
        return false;
    }
    else {
        cudaDeviceProp props;
        cudaStatus = cudaGetDeviceProperties(&props, DEVICE_NR);
        printf("Using:\n");
        printf("%s (%2d procs)\n", props.name, props.multiProcessorCount);
        printf("maxThreadsPerBlock: %2d\n\n", props.maxThreadsPerBlock);
        if (BLOCK_NUMBER == 0) {
            BLOCK_NUMBER = props.multiProcessorCount * 2;
        }
        if (BLOCK_THREADS == 0) {
            BLOCK_THREADS = props.maxThreadsPerBlock / 4;
        }
        outputSize = BLOCK_NUMBER * BLOCK_THREADS * THREAD_STEPS;
        loopStride = new Int(&STRIDE);
        loopStride.Mult(outputSize);
        useCollector = outputSize >= collectorLimit;
    }
    return true;
}

void showHelp() {    
    printf("WifSolverCuda [-d deviceId] [-b NbBlocks] [-t NbThreads] [-s NbThreadChecks]\n");
    printf("    [-fresultp reportFile] [-fresult resultFile] [-fstatus statusFile] [-a targetAddress]\n");
    printf("    -rangeStart hexKeyStart -rangeEnd hexKeyEnd -stride hexKeyStride\n\n");
    printf("-rangeStart hexKeyStart: decoded initial key with compression flag and checksum \n");
    printf("-rangeEnd hexKeyEnd: decoded end key with compression flag and checksum \n");
    printf("-stride hexKeyStride: full stride calculated as 58^(missing char index) \n");
    printf("-fresult resultFile: file for final result (default: %s)\n", fileResult.c_str());
    printf("-fresultp reportFile: file for each WIF with correct checksum (default: %s)\n", fileResultPartial.c_str());
    printf("-fstatus statusFile: file for periodically saved status (default: %s) \n", fileStatus.c_str());
    printf("-d deviceId: default 0\n");
    printf("-c search for compressed address\n");
    printf("-u search for uncompressed address (default)\n");
    printf("-b NbBlocks: default processorCount * 12\n");
    printf("-t NbThreads: default deviceMax / 4\n");
    printf("-s NbThreadChecks: default 3364\n");
    printf("-a targetAddress: expected address\n");    
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
        if (strcmp(argv[a], "-d") == 0) {
            a++;
            DEVICE_NR = strtol(argv[a], NULL, 10);
        }else
        if (strcmp(argv[a], "-c") == 0) {
            COMPRESSED = true;
        }
        else if (strcmp(argv[a], "-u") == 0) {
            COMPRESSED = false;
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
        else if (strcmp(argv[a], "-a") == 0) {
            a++;
            TARGET_ADDRESS = string(argv[a]);
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
