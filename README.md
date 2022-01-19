# WifSolverCuda
Tool for solving misspelled or damaged Bitcoin Private Key in Wallet Import Format (WIF)

Usage:

    WifSolverCuda [-d deviceId] [-b NbBlocks] [-t NbThreads] [-s NbThreadChecks]
         [-fresultp reportFile] [-fresult resultFile] [-fstatus statusFile] [-a targetAddress]
         -rangeStart hexKeyStart -rangeEnd hexKeyEnd -stride hexKeyStride     

     -rangeStart hexKeyStart: decoded initial key with compression flag and checksum
     -rangeEnd hexKeyEnd: decoded end key with compression flag and checksum
     -stride hexKeyStride: full stride calculated as 58^(missing char index)
     -fresult resultFile: file for final result (default: result.txt)
     -fresultp reportFile: file for each WIF with correct checksum (default: result_partial.txt)
     -fstatus statusFile: file for periodically saved status (default: fileStatus.txt)
     -d deviceId: default 0
     -b NbBlocks: default processorCount * 12
     -t NbThreads: default deviceMax / 4
     -s NbThreadChecks: default 3364
     -a targetAddress: expected address
     
Currently program supports only the case of uncompressed WIF with missing characters in the middle.
Support for compressed WIF and characters missing on the beginning will be added very soon.

Program could search for given address or search for any valid WIF with a given configuration. 
 
How to use it
-------------

In my examples I will use WIF _5KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKSmnqY_
which produces address _19NzcPZvZMSNQk8sDbSiyjeKpEVpaS1212_
The expected private key is: _c59cb0997ad73f7bf8621b1955caf80b304ded0a48e5b8f28c7b8f9356ec35e5_
    
Let's assume we have WIF with 5 missing characters _5KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK-----KKKSmnqY_
We must replace unknown characters by minimal characters from base58 encoding, to produce our starting key.
WIF _5KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK11111KKKSmnqY_ could be decoded to:
80c59cb0997ad73f7bf8621b1955caf80b304ded0a48e5b8f28c7b89f466ff5f68e2677283

In our calculations we need full decoded value, not only private key part.
The same way we may calculate maximum (end of range) for key processing - replacing unknown characters with 'z'.

Another important step is to calculate stride. Each change of unknown characters has impact on decoded value.
In our case the first missing character is on 9th position from right side, so our stride is
58^8 = 7479027ea100

We may launch program with parameters:

    -stride 7479027ea100 -rangeStart 80c59cb0997ad73f7bf8621b1955caf80b304ded0a48e5b8f28c7b89f466ff5f68e2677283  -a 19NzcPZvZMSNQk8sDbSiyjeKpEVpaS1212

Solver for described example is based on fact that stride modifies decoded checksum. Program verifies checksum (2*sha256) and only valid WIFs are checked agains expected address (pubkey->hashes->address).
    
For WIFs where stride does not collide with checksum, other algorithm will be needed.
        
Build
-----
Program was prepared using CUDA 11.6 - for other version manual change in VS project config files is needed.

Performance
-----------
One's must modify number of blocks and number of threads in each block to find the ones which are the best for his card. Number of test performed by each thread also could have impact of global performance/latency.  
Example: RTX3060 (-b 224 -t 512 -s 3364) checks around 1000Mkey/s. Example above extended to 7 missing characters was solved in 12 minutes (starting key: 80c59cb0997ad73f7bf8621b1955caf80b304ded0a48e5b8f28c31b30a90d68ffcabd9b283).
       
TODO
----
* code cleaning, review of hash functions
* build configuration for Linux
* support for compressed address
* solver for missing characters at the right side (with expected checksum)
* predefined step (using list of possible characters)
* reading configuration from file