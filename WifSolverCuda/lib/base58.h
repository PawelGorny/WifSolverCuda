#ifndef LIBBASE58_H
#define LIBBASE58_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

	extern bool (*b58_sha256_impl)(void*, const void*, size_t);
	extern bool b58enc(char* b58, size_t* b58sz, const void* bin, size_t binsz);
	extern bool b58tobin(void* bin, size_t* binsz, const char* b58, size_t b58sz);
#ifdef __cplusplus
}
#endif

#endif