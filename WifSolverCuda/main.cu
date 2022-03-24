
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
void listDevices();
void printConfig();
void printFooter();
void decodeWif();
void printSpeed(double speed);
void saveStatus();
void restoreSettings(string fileStatusRestore);

cudaError_t processCuda();


int DEVICE_NR = 0;
unsigned int BLOCK_THREADS = 0;
unsigned int BLOCK_NUMBER = 0;
unsigned int THREAD_STEPS = 1682;

size_t wifLen = 53;
int dataLen = 37;

bool COMPRESSED = false;
Int STRIDE, RANGE_START, RANGE_END, RANGE_START_TOTAL, RANGE_TOTAL;
double RANGE_TOTAL_DOUBLE;
Int loopStride;
Int counter;
string TARGET_ADDRESS = "";
Int CHECKSUM;
bool IS_CHECKSUM = false;

bool DECODE = false;
string WIF_TO_DECODE;

bool RESULT = false;
bool useCollector = false;
uint64_t collectorLimit = 100;

uint64_t outputSize;

string fileResultPartial = "result_partial.txt";
string fileResult = "result.txt";
string fileStatus = "fileStatus.txt";
int fileStatusInterval = 60;
string fileStatusRestore;
bool isRestore = false;

bool showDevices = false;
bool p2sh = false;

Secp256K1* secp;


int main(int argc, char** argv)
{    
    printf("WifSolver 0.4.8\n\n");

    if (readArgs(argc, argv)) {
        showHelp(); 
        printFooter();
        return 0;
    }
    if (showDevices) {
        listDevices();
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
    RANGE_START_TOTAL.Set(&RANGE_START);
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

    cudaError_t cudaStatus = processCuda();
    
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
        
    std::chrono::steady_clock::time_point beginCountHashrate = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point beginCountStatus = std::chrono::steady_clock::now();
   
    while (!RESULT && RANGE_START.IsLower(&RANGE_END)) {

        //prepare launch
        __Load(buffRangeStart, RANGE_START.bits64);
        cudaStatus = cudaMemcpy(dev_buffRangeStart, buffRangeStart, NB64BLOCK * sizeof(uint64_t), cudaMemcpyHostToDevice);
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
            cudaStatus = cudaMemcpy(buffCollectorWork, dev_buffCollectorWork, 1 * sizeof(bool), cudaMemcpyDeviceToHost);
            bool anyResult = buffCollectorWork[0];
            buffCollectorWork[0] = false;
            cudaStatus = cudaMemcpyAsync(dev_buffCollectorWork, buffCollectorWork, 1 * sizeof(bool), cudaMemcpyHostToDevice);
            if (anyResult) {
                for (int i = 0; i < COLLECTOR_SIZE; i++) {
                    buffResult[i] = 0;
                }
                cudaStatus = cudaMemcpy(dev_buffResult, buffResult, COLLECTOR_SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);
            }
            while (anyResult && !RESULT) {
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
                cudaStatus = cudaMemcpy(buffResult, dev_buffResult, COLLECTOR_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
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
            }
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
        int64_t tStatus = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - beginCountStatus).count();
        if (tHash > 5) {
            double speed = (double)((double)counter / tHash) / 1000000.0;
            printSpeed(speed);            
            counter = 0;
            beginCountHashrate = std::chrono::steady_clock::now();
        }
        if (tStatus > fileStatusInterval) {
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
            TARGET_ADDRESS = s.substr(prefix.size(), s.size() - prefix.size() - 1);
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
    fprintf(stat, "-rangeStart=%s\n", RANGE_START.GetBase16().c_str());
    fprintf(stat, "-rangeEnd=%s\n", RANGE_END.GetBase16().c_str());
    fprintf(stat, "-stride=%s\n", STRIDE.GetBase16().c_str());
    if (!TARGET_ADDRESS.empty()) {
        fprintf(stat, "-a=%s\n", TARGET_ADDRESS.c_str());
    }else {
        fprintf(stat, "-a=\n");
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
    printf("\r %s,  progress: %.3f%%     ", speedStr.c_str(), _count);    
    //printf("\r %s,    ", speedStr.c_str()); 
    fflush(stdout);
}

void processCandidate(Int &toTest) {     
    FILE* keys;
    char rmdhash[21], address[50], wif[53];        
    unsigned char* buff = new unsigned char[dataLen];
    for (int i = 0, d=dataLen-1; i < dataLen; i++, d--) {
        buff[i] = toTest.GetByte(d);
    }       
    toTest.SetBase16((char*)toTest.GetBase16().substr(2, 64).c_str());        
    Point publickey = secp->ComputePublicKey(&toTest);        
    if (p2sh) {
        secp->GetHash160(P2SH, true, publickey, (unsigned char*)rmdhash);
    }
    else {
        secp->GetHash160(P2PKH, COMPRESSED, publickey, (unsigned char*)rmdhash);
    }
    addressToBase58(rmdhash, address, p2sh);    
    if (!TARGET_ADDRESS.empty()) {
        if (TARGET_ADDRESS == address) {
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

void printConfig() {
    printf("Range start: %s\n", RANGE_START.GetBase16().c_str());
    printf("Range end  : %s\n", RANGE_END.GetBase16().c_str());
    printf("Stride     : %s\n", STRIDE.GetBase16().c_str());
    if (!TARGET_ADDRESS.empty()) {
        printf( "Target     : %s\n", TARGET_ADDRESS.c_str());
    }
    if (COMPRESSED) {
        printf("Target COMPRESSED\n");
    }    else    {
        printf("Target UNCOMPRESSED\n");
    }   
    if (IS_CHECKSUM) {
        printf("Checksum   : %s\n", CHECKSUM.GetBase16().c_str());
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
            BLOCK_NUMBER = props.multiProcessorCount * 8;
        }
        if (BLOCK_THREADS == 0) {
            BLOCK_THREADS = (props.maxThreadsPerBlock / 8) * 5;
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
    printf("    -stride hexKeyStride -rangeStart hexKeyStart [-rangeEnd hexKeyEnd] [-checksum hexChecksum] \n");
    printf("    [-decode wifToDecode] \n");
    printf("    [-restore statusFile] \n");
    printf("    [-listDevices] \n");
    printf("    [-h] \n\n");
    printf("-rangeStart hexKeyStart: decoded initial key with compression flag and checksum \n");
    printf("-rangeEnd hexKeyEnd:     decoded end key with compression flag and checksum \n");
    printf("-checksum hexChecksum:   decoded checksum, cannot be modified with a stride  \n");
    printf("-stride hexKeyStride:    full stride calculated as 58^(most-right missing char index) \n");
    printf("-a targetAddress:        expected address\n");
    printf("-fresult resultFile:     file for final result (default: %s)\n", fileResult.c_str());
    printf("-fresultp reportFile:    file for each WIF with correct checksum (default: %s)\n", fileResultPartial.c_str());
    printf("-fstatus statusFile:     file for periodically saved status (default: %s) \n", fileStatus.c_str());
    printf("-fstatusIntv seconds:    period between status file updates (default %d sec) \n", fileStatusInterval);
    printf("-d deviceId:             default 0\n");
    printf("-c :                     search for compressed address\n");
    printf("-u :                     search for uncompressed address (default)\n");
    printf("-b NbBlocks:             default processorCount * 8\n");
    printf("-t NbThreads:            default deviceMax/8 * 5\n");
    printf("-s NbThreadChecks:       default %d\n", THREAD_STEPS);
    printf("-decode wifToDecode:     decodes given WIF\n");    
    printf("-restore statusFile:     restore work configuration\n");
    printf("-listDevices:            shows available devices\n");
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
            DEVICE_NR = strtol(argv[a], NULL, 10);
        }
        else if (strcmp(argv[a], "-c") == 0) {
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
        else if (strcmp(argv[a], "-fstatusIntv") == 0) {
            a++;
            fileStatusInterval = strtol(argv[a], NULL, 10);
        }
        else if (strcmp(argv[a], "-a") == 0) {
            a++;
            TARGET_ADDRESS = string(argv[a]);
            if (argv[a][0] == '3') {
                p2sh = true;
            }
        }
        else if (strcmp(argv[a], "-checksum") == 0) {
            a++;
            CHECKSUM.SetBase16((char*)string(argv[a]).c_str());
            IS_CHECKSUM = true;
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
        printf("  %2d procs\n", prop.multiProcessorCount);
        printf("  maxThreadsPerBlock: %2d\n\n", prop.maxThreadsPerBlock);
    }
}