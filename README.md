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
     -c search for compressed address
     -u search for uncompressed address (default)
     -d deviceId: default 0
     -b NbBlocks: default processorCount * 12
     -t NbThreads: default deviceMax / 4
     -s NbThreadChecks: default 3364
     -a targetAddress: expected address
     
Currently program supports only the case of compressed or uncompressed WIF with missing characters in the middle.
Support for WIFs with characters missing on the beginning will be added very soon.

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

    -stride 7479027ea100 -u -rangeStart 80c59cb0997ad73f7bf8621b1955caf80b304ded0a48e5b8f28c7b89f466ff5f68e2677283  -a 19NzcPZvZMSNQk8sDbSiyjeKpEVpaS1212

Solver for described example is based on fact that stride modifies decoded checksum. Program verifies checksum (2*sha256) and only valid WIFs are checked agains expected address (pubkey->hashes->address).
    
Similar test for compressed WIF (target _KzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzWK7YY3s_):

    -stride 7479027ea100 -c -rangeStart 8070cfa0d40309798a5bd144a396478b5b5ae3305b7413601b18767654f1108a02787692623a  -a 1PzaLTZS3J3HqGfsa8Z2jfkCT1QpSMVunD
   
For WIFs where stride does not collide with checksum, other algorithm will be needed.
        
Build
-----
Program was prepared using CUDA 11.6 - for other version manual change in VS project config files is needed.

Performance
-----------
One's must modify number of blocks and number of threads in each block to find the ones which are the best for his card. Number of test performed by each thread also could have impact of global performance/latency.  
Test card: RTX3060 (-b 224 -t 512 -s 3364) checks around 3600 MKey/s for compressed address with missing characters in the middle and around 1300Mkey/s for other cases.

       
TODO
----
* code cleaning, review of hash functions
* build configuration for Linux
* predefined custom step (using list of possible characters)
* reading configuration from file
* build-in stride calculcater