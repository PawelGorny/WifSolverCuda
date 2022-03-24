#include <iostream>

#ifndef CUSTOMUTILH
#define CUSTOMUTILH

typedef struct str_list {
	int n;
	char** data;
	int* lengths;
}List;

typedef struct str_tokenizer {
	int current;
	int n;
	char** tokens;
}Tokenizer;

char* ltrim(char* str, const char* seps);
char* rtrim(char* str, const char* seps);
char* trim(char* str, const char* seps);
int indexOf(char* s, const char** array, int length_array);

int hexchr2bin(char hex, char* out);
int hexs2bin(char* hex, unsigned char* out);
char* tohex(char* ptr, int length);
void tohex_dst(char* ptr, int length, char* dst);

int hasMoreTokens(Tokenizer* t);
char* nextToken(Tokenizer* t);

int isValidHex(char* data);
void freetokenizer(Tokenizer* t);
void stringtokenizer(char* data, Tokenizer* t);

void addressToBase58(char* rmd, char* dst, bool p2sh);

std::string formatDouble(const char* formatStr, double value);

bool b58encode(char* b58, size_t* b58sz, const void* data, size_t binsz);
bool b58decode(unsigned char* bin, size_t* binszp, const char* b58, size_t b58sz);

#endif // CUSTOMUTILH
