# WifSolverCuda
Tool for solving misspelled or damaged Bitcoin Private Key in Wallet Import Format (WIF)

Usage:

    WifSolverCuda [-d deviceId] [-b NbBlocks] [-t NbThreads] [-s NbThreadChecks]
         [-fresultp reportFile] [-fresult resultFile] [-fstatus statusFile] [-a targetAddress]
         -stride hexKeyStride -rangeStart hexKeyStart [-rangeEnd hexKeyEnd] [-checksum hexChecksum] 
         [-decode wifToDecode]
         [-restore statusFile]
         [-listDevices] [-disable-um]
         [-h]

     -rangeStart hexKeyStart: decoded initial key with compression flag and checksum
     -rangeEnd hexKeyEnd:     decoded end key with compression flag and checksum
     -stride hexKeyStride:    full stride calculated as 58^(most-right missing char index)
	 -checksum hexChecksum:   decoded checksum, cannot be modified with a stride
	 -a targetAddress:        expected address
     -fresult resultFile:     file for final result (default: result.txt)
     -fresultp reportFile:    file for each WIF with correct checksum (default: result_partial.txt)
     -fstatus statusFile:     file for periodically saved status (default: fileStatus.txt)
     -fstatusIntv seconds:    period between status file updates (default 60 sec)
	 -d deviceId:             default 0
     -c :                     search for compressed address
     -u :                     search for uncompressed address (default)     
     -b NbBlocks:             default processorCount * 8
     -t NbThreads:            default deviceMax/8 * 5
     -s NbThreadChecks:       default 3364
     -decode wifToDecode:     decodes given WIF
     -restore statusFile:     restore work configuration
     -listDevices:            shows available devices
     -disable-um:             disable unified memory mode
     -h :                     shows help
     

Program could search for given address or search for any valid WIF with a given configuration. 
 
How to use it
-------------

In my examples I will use WIF _5KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKSmnqY_
which produces address _19NzcPZvZMSNQk8sDbSiyjeKpEVpaS1212_
The expected private key is: _c59cb0997ad73f7bf8621b1955caf80b304ded0a48e5b8f28c7b8f9356ec35e5_
    
Let's assume we have WIF with 5 missing characters _5KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK?????KKKSmnqY_
We must replace unknown characters by minimal characters from base58 encoding, to produce our starting key.
WIF _5KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK11111KKKSmnqY_ could be decoded to:
80c59cb0997ad73f7bf8621b1955caf80b304ded0a48e5b8f28c7b89f466ff5f68e2677283

In our calculations we need full decoded value, not only private key part.
The same way we may calculate maximum (end of range) for key processing - replacing unknown characters with 'z'.

Another important step is to calculate stride. Each change of unknown characters has impact on decoded value.
In our case the first missing character is on 9th position from right side, so our stride is
58^8 = 7479027ea100

We may launch program with parameters:

    -stride 7479027ea100 -u -rangeStart 80c59cb0997ad73f7bf8621b1955caf80b304ded0a48e5b8f28c7b89f466ff5f68e2677283  -a 19NzcPZvZMSNQk8sDbSiyjeKpEVpaS1212

Solver for described example is based on fact that stride modifies decoded checksum. Program verifies checksum (2*sha256) and only valid WIFs are checked agains expected address (pubkey->hashes->address).
    
Similar test for compressed WIF (target _KzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzWK7YY3s_):

    -stride 7479027ea100 -c -rangeStart 8070cfa0d40309798a5bd144a396478b5b5ae3305b7413601b18767654f1108a02787692623a  -a 1PzaLTZS3J3HqGfsa8Z2jfkCT1QpSMVunD
   
For other of examples please see the file /docs/examples.txt. 

        
Build
-----
Windows:

Program was prepared using CUDA 11.6 - for any other version manual change in VS project config files is needed. Exe under /Releases/ was build using compute_cap=86, for cards 30xx. If you have older card, you must rebuild program using older CUDA/lower CCAP.

Linux:

Go to WifSolverCuda/ subfolder and execute _make all_. If your device does not support compute capability=86 (error "No kernel image is available for execution on the device"), do the change in _Makefile_ (for example 1080Ti requires COMPUTE_CAP=61).


Performance
-----------
User should modify number of blocks and number of threads in each block to find values which are the best for his card. Number of tests performed by each thread also could have impact of global performance/latency.  

Test card: RTX3060 (eGPU!) with 224 BLOCKS & 512 BLOCK_THREADS (program default values) checks around 10000 MKey/s for compressed address with missing characters in the middle (collision with checksum) and around 1400-1540 Mkey/s for missing beginning (20000steps/thread); other results (using default values of blocks, threads and steps per thread):

| card          | perf Mkey/s, missing beginning
|---------------|---------------------|
| RTX 3060 eGPU | 1520 (224/512/20000)|
| RTX 3070      | 2200 (414/640/5000) |
| RTX 3090      | 3950 (656/640/5000) |
| RTX 3080TI    | 4090 (640/640/5000) |
| RTX A6000     | 4070 (588/640/5000) |
| GTX 1070      | 950  (135/768/5000) |

Please consult official Nvidia Occupancy Calculator (https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html) to see how to select desired amount of threads/block (shared memory=0, registers per thread = 48). Adjust number of steps per thread to obtain the optimal performance.
       
TODO
----
* code cleaning, review of hash functions
* predefined custom step (using list of possible characters)
* auto-processing (preparing configuration) based on WIF

Contact
-------
Contact email: pawgo@protonmail.com
If you found this program useful, consider making a donation, I will appreciate it! 
**BTC**: `bc1qz2akvlch75rqdfg8pv7chqvz3m8jsl49k0kszc`
