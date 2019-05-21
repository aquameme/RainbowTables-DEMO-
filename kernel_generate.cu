#pragma once

// CUDA includes.
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_defines.h"
#include "kernel_generate.cuh"

// C includes.
#include <stdint.h>

/* SHA-256 constants. */
// Host.

 // Device
__device__ __constant__ uint32_t sha256_h[8] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
	0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};
__device__ __constant__ uint32_t sha256_k[64] = {
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
		0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
		0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
		0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
		0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
		0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
		0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
		0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
		0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
		0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
		0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
		0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

/* Elementary functions used by SHA-256. */

/*
#define Ch(x, y, z)     ((x & (y ^ z)) ^ z)
#define Maj(x, y, z)    ((x & (y | z)) | (y & z))

#define Ch(x, y, z)		((x & y) ^ (~(x) & z))
#define Maj(x, y, z)	((x & y) ^ (x & z) ^ (y & z))
*/
#define Ch(x, y, z)     ((x & (y ^ z)) ^ z)
#define Maj(x, y, z)    ((z & (x | y)) | (x & y))
#define ROTR(x, n)      ((x >> n) | (x << (32 - n)))
#define S0(x)           (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define S1(x)           (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define s0(x)           (ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3))
#define s1(x)           (ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10))

/*
* Description:
* SHA256 round function (without final shift).
* wk = W[i] + K[i].
*/
#define RND(a, b, c, d, e, f, g, h, wk) \
    do { \
        t0 = h + S1(e) + Ch(e, f, g) + wk; \
        t1 = S0(a) + Maj(a, b, c); \
		h = g; \
		g = f; \
		f = e; \
		e = d + t0; \
		d = c; \
		c = b; \
		b = a; \
		a = t0 + t1; \
			    } while (0)

/*
* Description:
* Adjusted round function for rotating state.
*/
#define RNDr(S, W, i) \
    RND(S[0 * SHARED_MEMORY_BANK_COUNT], S[1 * SHARED_MEMORY_BANK_COUNT], \
        S[2 * SHARED_MEMORY_BANK_COUNT], S[3 * SHARED_MEMORY_BANK_COUNT], \
        S[4 * SHARED_MEMORY_BANK_COUNT], S[5 * SHARED_MEMORY_BANK_COUNT], \
        S[6 * SHARED_MEMORY_BANK_COUNT], S[7 * SHARED_MEMORY_BANK_COUNT], \
        W[i] + sha256_k[i])


/* Functions. */

__device__ void sha256_copy_hash(uint32_t *threadHash_shared, uint32_t *threadHash_global)
{
#pragma unroll 8
	for (int i = 0; i < 8; i++)
	{
		threadHash_shared[i * SHARED_MEMORY_BANK_COUNT] = threadHash_global[i];
	}
}

__device__ void sha256_init_hash(uint32_t *threadHash_shared)
{
#pragma unroll 8
	for (int i = 0; i < 8; i++)
	{
		threadHash_shared[i * SHARED_MEMORY_BANK_COUNT] = sha256_h[i];
	}
}

__device__ void sha256_init_pass(char *threadBlock_shared, char *threadBlock_global, const int maxPasswordLength)
{
	for (int i = 0; i < maxPasswordLength; i++)
	{
		threadBlock_shared[(i / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + (i % SHARED_MEMORY_BANK_WIDTH)] = threadBlock_global[i];
	}
}

__device__ void make_full_block(char *threadBlock_shared, const int maxPasswordLength)
{
	// Password is <= 32 symbols (possibly to expand to 56 symbols)
	// First of all, copy password bits to the start of the block. - done by default
	// Then add a '1' bit after the password.

	int passwordLength = 0;
	for (int j = 0; (threadBlock_shared[j] != '\0') && (passwordLength < maxPasswordLength); passwordLength++)
	{
		j = (passwordLength + 1) % SHARED_MEMORY_BANK_WIDTH + ((passwordLength + 1) / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH;
	}
	//passwordLength--;

	threadBlock_shared[(passwordLength / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + passwordLength % SHARED_MEMORY_BANK_WIDTH] = (char)0x80; // binary - 10000000

	// Leave all other bits to be '0', except last 64.

	for (int i = passwordLength + 1; i < 56; i++)
		threadBlock_shared[(i / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + i % SHARED_MEMORY_BANK_WIDTH] = '\0'; // 00000000

	// Now data is written in little endian -> convert to big endian.

#pragma unroll 14
	for (int i = 0; i < 14; i++)
	{
		// For 16 groups of 4 bytes (reverse each uint32_t in block, except length - it's already written as uint64_t).

		char temp;

		// Exchange lower and higher bytes.

		temp = threadBlock_shared[((4 * i) / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + (4 * i) % SHARED_MEMORY_BANK_WIDTH];
		threadBlock_shared[((4 * i) / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + (4 * i) % SHARED_MEMORY_BANK_WIDTH]
			= threadBlock_shared[((4 * i + 3) / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + (4 * i + 3) % SHARED_MEMORY_BANK_WIDTH];
		threadBlock_shared[((4 * i + 3) / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + (4 * i + 3) % SHARED_MEMORY_BANK_WIDTH] = temp;

		// Exchange middle bytes.

		temp = threadBlock_shared[((4 * i + 1) / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + (4 * i + 1) % SHARED_MEMORY_BANK_WIDTH];
		threadBlock_shared[((4 * i + 1) / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + (4 * i + 1) % SHARED_MEMORY_BANK_WIDTH]
			= threadBlock_shared[((4 * i + 2) / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + (4 * i + 2) % SHARED_MEMORY_BANK_WIDTH];
		threadBlock_shared[((4 * i + 2) / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + (4 * i + 2) % SHARED_MEMORY_BANK_WIDTH] = temp;
	}

	// Write password length (in bits!) to last 64 bits (last 8 chars).
	// Passwords are less than 62 chars (496 bits) -> length will suit in uint32_t; previous 4 bytes will be 0.

	uint32_t *lastBlock64bits = (uint32_t *)&(threadBlock_shared[(56 / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + 56 % SHARED_MEMORY_BANK_WIDTH]);
	lastBlock64bits[0] = 0;
	lastBlock64bits[SHARED_MEMORY_BANK_COUNT] = (uint32_t)(passwordLength * 8);
}

__device__ void sha256_transform(uint32_t *threadHash_shared, const uint32_t *threadBlock_shared)
{
	uint32_t W[64]; // only 4 of these are accessed during each partial Mix
	uint32_t *S = threadHash_shared;
	uint32_t t0, t1;
	int i;

	/* 1. Prepare message schedule W */
#pragma unroll 16
	for (int i = 0; i < 16; i++) // WARNING! was 64< is it wrong?
		W[i] = threadBlock_shared[i * SHARED_MEMORY_BANK_COUNT];

	/* 2. Hash! */
	//mycpy16(W, block);
	RNDr(S, W, 0);
	RNDr(S, W, 1);
	RNDr(S, W, 2);
	RNDr(S, W, 3);

	//mycpy16(W + 4, block + 4);
	RNDr(S, W, 4); RNDr(S, W, 5); RNDr(S, W, 6); RNDr(S, W, 7);

	//mycpy16(W + 8, block + 8);
	RNDr(S, W, 8); RNDr(S, W, 9); RNDr(S, W, 10); RNDr(S, W, 11);

	//mycpy16(W + 12, block + 12);
	RNDr(S, W, 12); RNDr(S, W, 13); RNDr(S, W, 14); RNDr(S, W, 15);

#pragma unroll 2
	for (i = 16; i < 20; i += 2)
	{
		W[i] = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
		W[i + 1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15];
	}
	RNDr(S, W, 16); RNDr(S, W, 17); RNDr(S, W, 18); RNDr(S, W, 19);

#pragma unroll 2
	for (i = 20; i < 24; i += 2)
	{
		W[i] = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
		W[i + 1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15];
	}
	RNDr(S, W, 20); RNDr(S, W, 21); RNDr(S, W, 22); RNDr(S, W, 23);

#pragma unroll 2
	for (i = 24; i < 28; i += 2)
	{
		W[i] = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
		W[i + 1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15];
	}
	RNDr(S, W, 24); RNDr(S, W, 25); RNDr(S, W, 26); RNDr(S, W, 27);

#pragma unroll 2
	for (i = 28; i < 32; i += 2)
	{
		W[i] = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
		W[i + 1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15];
	}
	RNDr(S, W, 28); RNDr(S, W, 29); RNDr(S, W, 30); RNDr(S, W, 31);

#pragma unroll 2
	for (i = 32; i < 36; i += 2)
	{
		W[i] = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
		W[i + 1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15];
	}
	RNDr(S, W, 32); RNDr(S, W, 33); RNDr(S, W, 34); RNDr(S, W, 35);

#pragma unroll 2
	for (i = 36; i < 40; i += 2)
	{
		W[i] = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
		W[i + 1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15];
	}
	RNDr(S, W, 36); RNDr(S, W, 37); RNDr(S, W, 38); RNDr(S, W, 39);

#pragma unroll 2
	for (i = 40; i < 44; i += 2)
	{
		W[i] = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
		W[i + 1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15];
	}
	RNDr(S, W, 40); RNDr(S, W, 41); RNDr(S, W, 42); RNDr(S, W, 43);

#pragma unroll 2
	for (i = 44; i < 48; i += 2)
	{
		W[i] = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
		W[i + 1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15];
	}
	RNDr(S, W, 44); RNDr(S, W, 45); RNDr(S, W, 46); RNDr(S, W, 47);

#pragma unroll 2
	for (i = 48; i < 52; i += 2)
	{
		W[i] = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
		W[i + 1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15];
	}
	RNDr(S, W, 48); RNDr(S, W, 49); RNDr(S, W, 50); RNDr(S, W, 51);

#pragma unroll 2
	for (i = 52; i < 56; i += 2)
	{
		W[i] = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
		W[i + 1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15];
	}
	RNDr(S, W, 52); RNDr(S, W, 53); RNDr(S, W, 54); RNDr(S, W, 55);

#pragma unroll 2
	for (i = 56; i < 60; i += 2)
	{
		W[i] = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
		W[i + 1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15];
	}
	RNDr(S, W, 56); RNDr(S, W, 57); RNDr(S, W, 58); RNDr(S, W, 59);

#pragma unroll 2
	for (i = 60; i < 64; i += 2)
	{
		W[i] = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
		W[i + 1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15];
	}
	RNDr(S, W, 60); RNDr(S, W, 61); RNDr(S, W, 62); RNDr(S, W, 63);

	/* 3. Mix local working variables into global state - IN CASE OF MULTI-BLOCK MESSAGE */
	/*    In case of rainbow tables, passwords fit in 1 block -> no need to mix with global state. */
	// Instead just add sha256_h constants to ready hash.
#pragma unroll 8
	for (i = 0; i < 8; i++)
	{
		threadHash_shared[i * SHARED_MEMORY_BANK_COUNT] += sha256_h[i];
	}
}

__device__ void reduction_human(uint32_t *threadHash_shared, const int minPasswordLength, const int maxPasswordLength,
	uint32_t iterationNumber, const uint16_t *mtableStatsNums_global, const uint8_t *mtableStatsRating_global,
	const char *mtableChar_global, char *threadBlock_shared, uint32_t padding, uint8_t humanity)
{
	// cut due to the lack of an objective assessment of the importance of developing
}

__device__ void reduction_rand(uint32_t *threadHash_shared, const int minPasswordLength, const int maxPasswordLength,
	uint32_t iterationNumber, char *threadBlock_shared, uint32_t padding)
{
#pragma unroll 8
	for (uint8_t i = 0; i < 8; i++)
		threadHash_shared[i * SHARED_MEMORY_BANK_COUNT] ^= (iterationNumber + padding);

	char *threadHash_byteptr_shared = (char *)threadHash_shared;

	uint8_t passLength = ((uint8_t)threadHash_byteptr_shared[0] % (uint8_t)(maxPasswordLength - minPasswordLength + 1)) + (uint8_t)minPasswordLength;
	char this_byte;

	/*GET PASSWORD FROM HASH*/
	
	for (uint8_t i = 0; i < passLength; i++)
	{
		this_byte = threadHash_byteptr_shared[((i + 1) / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + (i + 1) % SHARED_MEMORY_BANK_WIDTH];
		threadBlock_shared[(i / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + i % SHARED_MEMORY_BANK_WIDTH] = (char)((uint8_t)this_byte % 95 + 32);
	}	

	/*ADD '\0's AT THE END OF PASSWORD TO READ CORRECTLY*/

	for (uint8_t i = passLength; i < maxPasswordLength; i++)
		threadBlock_shared[(i / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + i % SHARED_MEMORY_BANK_WIDTH] = '\0';
}

__global__ void rainbowKernel_generate(const int minPasswordLength, const int maxPasswordLength,
	const uint16_t *mtableStatsNums_global, const uint8_t *mtableStatsRating_global, const char *mtableChar_global, 
	const int chainLength, char *plainBlock_global, bool mode, uint32_t padding, uint8_t humanity)
{
	/* Shared memory */

	extern __shared__ uint32_t sharedMemPtr[];

	// Allocate memory for all threads' hash states.
	/*
	BANK NUMBER
	0   1   2   ...  30  31
	a0  a1  a2  ...  a30 a31
	b0  b1  b2  ...  b30 b31
	........................
	h0  h1  h2  ...  h30 h31
	a32 a33 a34 ...  a62 a63
	b32 b33 b34 ...  b62 b63
	........................

	a0-h0 - state for thread 0.
	And so on...
	*/
	// Hashes take blockDim.x * 8  4-byte cells = 16 KB.

	uint32_t *hash_shared = sharedMemPtr;
	uint32_t *threadHash_shared = &(hash_shared[(threadIdx.x / SHARED_MEMORY_BANK_COUNT) *
		(8 * SHARED_MEMORY_BANK_COUNT) + threadIdx.x % SHARED_MEMORY_BANK_COUNT]);

	// Allocate memory for 512-bit-size block (same as threadPassword_shared). Scheme is identical to states.
	/*
		BANK NUMBER
	1     2     ...  32
	pass  pass  ...  pass
	word  word  ...  word
	0XXX  1XXX  ...  31XX
	pass  pass  ...  pass
	word  word  ...  word
	32XX  33XX  ...  63XX
	.....................

	1 symbol - 1 byte
	X - empty, unused byte
	*/
	uint16_t normalizedBlockDim;
	if (blockDim.x % 32)
		normalizedBlockDim = (blockDim.x / 32 + 1) * 32;
	else
		normalizedBlockDim = blockDim.x;

	uint32_t *blocks_shared = &hash_shared[normalizedBlockDim * 8];
	uint32_t *threadBlock_shared = &(blocks_shared[(threadIdx.x / SHARED_MEMORY_BANK_COUNT) * 
		(16 * SHARED_MEMORY_BANK_COUNT) + threadIdx.x % SHARED_MEMORY_BANK_COUNT]);
	char *threadBlock_byteptr_shared = (char *)threadBlock_shared;

	// Allocate pointers for chain starts

	char *threadBlock_global = &(plainBlock_global[(blockIdx.x * blockDim.x + threadIdx.x) * maxPasswordLength]);
	
	// Example: 
	// blockDim = 512
	// Block #0 will take passwords from 1 to 512
	// Block #1 will take passwords from 513 to 1024
	// Block #3 will take passwords from 1025 to 1536
	// ...

	// Addressing: 
	// (threadIdx.x / 32) * (((maxPasswordLength - 1) / 4 + 1) * 128) + (threadIdx.x % 32) * 4
	//             |                                    |         |					  |     |
	//  line block number		          			 bank width  line size    bank width    bank width

	// Generate chains!

	sha256_init_pass(threadBlock_byteptr_shared, threadBlock_global, maxPasswordLength);

	__syncthreads();

	for (uint32_t i = 0; i < chainLength; i++)
	{ //  A password -> hash -> password iteration.
		/**passwordLength = 3;
		threadBlock_shared[0] = 'a';
		threadBlock_shared[1] = 'b';
		threadBlock_shared[2] = 'c';*/

		// Transform password into a valid for sha256 512-bit block.
		make_full_block(threadBlock_byteptr_shared, maxPasswordLength);

		// Hash the password.
		sha256_init_hash(threadHash_shared);
		sha256_transform(threadHash_shared, (const uint32_t *)threadBlock_shared);
		// Now state contains a sha256 hash of the password.

		// Perform reduction.

		if (mode)
			reduction_human(threadHash_shared, minPasswordLength, maxPasswordLength, i, mtableStatsNums_global, mtableStatsRating_global, mtableChar_global, threadBlock_byteptr_shared, padding, humanity);
		else
			reduction_rand(threadHash_shared, minPasswordLength, maxPasswordLength, i, threadBlock_byteptr_shared, padding);

		// Now threadBlock_shared contains a password made with reduction
	}

	/*READ ENDS FROM SHARED MEMORY*/
	
	for (int i = 0, j = 0; i < maxPasswordLength; i++)
	{
		threadBlock_global[i] = threadBlock_byteptr_shared[j];
		j = (i + 1) % SHARED_MEMORY_BANK_WIDTH + ((i + 1) / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH;
	}
}

__global__ void rainbowKernel_probability(const int minPasswordLength, const int maxPasswordLength,
	const uint16_t *mtableStatsNums_global, const uint8_t *mtableStatsRating_global, const char *mtableChar_global,
	const int chainLength, char *plainBlock_global, uint32_t *hashBlock_global, uint64_t calculated, bool mode, uint32_t padding, uint8_t humanity)
{
	/* Shared memory */

	extern __shared__ uint32_t sharedMemPtr[];

	uint32_t *hash_shared = sharedMemPtr;
	uint32_t *threadHash_shared = &(hash_shared[(threadIdx.x / SHARED_MEMORY_BANK_COUNT) *
		(8 * SHARED_MEMORY_BANK_COUNT) + threadIdx.x % SHARED_MEMORY_BANK_COUNT]);

	// Allocate memory for 512-bit-size block (same as threadPassword_shared). Scheme is identical to states.

	uint16_t normalizedBlockDim;
	if (blockDim.x % 32)
		normalizedBlockDim = (blockDim.x / 32 + 1) * 32;
	else
		normalizedBlockDim = blockDim.x;

	uint32_t *blocks_shared = &hash_shared[normalizedBlockDim * 8];
	uint32_t *threadBlock_shared = &(blocks_shared[(threadIdx.x / SHARED_MEMORY_BANK_COUNT) *
		(16 * SHARED_MEMORY_BANK_COUNT) + threadIdx.x % SHARED_MEMORY_BANK_COUNT]);
	char *threadBlock_byteptr_shared = (char *)threadBlock_shared;

	// Allocate pointers for chain starts

	char *threadBlock_global = &(plainBlock_global[(blockIdx.x * blockDim.x + threadIdx.x) * maxPasswordLength]);
	uint32_t *threadHash_global = &(hashBlock_global[((calculated + blockIdx.x * blockDim.x + threadIdx.x) / chainLength) * 8]);
	uint16_t localChainLength = (calculated + (blockIdx.x * blockDim.x + threadIdx.x)) % chainLength;

	// Generate chains!

	/* copy hashes */

	sha256_copy_hash(threadHash_shared, threadHash_global);

	//__syncthreads();

	for (uint32_t i = chainLength - localChainLength - 1; i < chainLength; i++)
	{ 
			// A hash -> password -> hash iteration.

		if (i != (chainLength - localChainLength - 1))
		{
			// Transform password into a valid for sha256 512-bit block.
			make_full_block(threadBlock_byteptr_shared, maxPasswordLength);

			// Hash the password.
			sha256_init_hash(threadHash_shared);
			sha256_transform(threadHash_shared, (const uint32_t *)threadBlock_shared);
			// Now state contains a sha256 hash of the password.
		}

		// Perform reduction.

		if (mode)
			reduction_human(threadHash_shared, minPasswordLength, maxPasswordLength, i, mtableStatsNums_global, mtableStatsRating_global, mtableChar_global, threadBlock_byteptr_shared, padding, humanity);
		else
			reduction_rand(threadHash_shared, minPasswordLength, maxPasswordLength, i, threadBlock_byteptr_shared, padding);

		// Now threadBlock_shared contains a password made with reduction
	}

	/*READ ENDS FROM SHARED MEMORY*/

	for (int i = 0, j = 0; i < maxPasswordLength; i++)
	{
		threadBlock_global[i] = threadBlock_byteptr_shared[j];
		j = (i + 1) % SHARED_MEMORY_BANK_WIDTH + ((i + 1) / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH;
	}
}

__global__ void rainbowKernel_expr_probability(const int minPasswordLength, const int maxPasswordLength,
	const uint16_t *mtableStatsNums_global, const uint8_t *mtableStatsRating_global, const char *mtableChar_global,
	const int chainLength, uint64_t chainAmount, char *plainBlock_global, char *RTStartsBlock_global, char *RTEndsBlock_global,
	uint32_t *hashBlock_global, uint64_t calculated, bool mode, uint32_t padding, uint8_t humanity)
{
	/* Shared memory */

	extern __shared__ uint32_t sharedMemPtr[];

	uint32_t *hash_shared = sharedMemPtr;
	uint32_t *threadHash_shared = &(hash_shared[(threadIdx.x / SHARED_MEMORY_BANK_COUNT) *
		(8 * SHARED_MEMORY_BANK_COUNT) + threadIdx.x % SHARED_MEMORY_BANK_COUNT]);

	// Allocate memory for 512-bit-size block (same as threadPassword_shared). Scheme is identical to states.

	uint16_t normalizedBlockDim;
	if (blockDim.x % 32)
		normalizedBlockDim = (blockDim.x / 32 + 1) * 32;
	else
		normalizedBlockDim = blockDim.x;

	uint32_t *blocks_shared = &hash_shared[normalizedBlockDim * 8];
	uint32_t *threadBlock_shared = &(blocks_shared[(threadIdx.x / SHARED_MEMORY_BANK_COUNT) *
		(16 * SHARED_MEMORY_BANK_COUNT) + threadIdx.x % SHARED_MEMORY_BANK_COUNT]);
	char *threadBlock_byteptr_shared = (char *)threadBlock_shared;

	// Allocate pointers for chain starts

	char *threadBlock_global = &(plainBlock_global[(blockIdx.x * blockDim.x + threadIdx.x) * maxPasswordLength]);
	uint32_t *threadHash_global = &(hashBlock_global[((calculated + blockIdx.x * blockDim.x + threadIdx.x) / chainLength) * 8]);
	uint16_t localChainLength = (calculated + (blockIdx.x * blockDim.x + threadIdx.x)) % chainLength;

	// Generate chains!

	/* copy hashes */

	sha256_copy_hash(threadHash_shared, threadHash_global);

	//__syncthreads();

	for (uint32_t i = chainLength - localChainLength - 1; i < chainLength; i++)
	{
		// A hash -> password -> hash iteration.

		if (i != (chainLength - localChainLength - 1))
		{
			// Transform password into a valid for sha256 512-bit block.
			make_full_block(threadBlock_byteptr_shared, maxPasswordLength);

			// Hash the password.
			sha256_init_hash(threadHash_shared);
			sha256_transform(threadHash_shared, (const uint32_t *)threadBlock_shared);
			// Now state contains a sha256 hash of the password.
		}

		// Perform reduction.

		if (mode)
			reduction_human(threadHash_shared, minPasswordLength, maxPasswordLength, i, mtableStatsNums_global, mtableStatsRating_global, mtableChar_global, threadBlock_byteptr_shared, padding, humanity);
		else
			reduction_rand(threadHash_shared, minPasswordLength, maxPasswordLength, i, threadBlock_byteptr_shared, padding);

		// Now threadBlock_shared contains a password made with reduction
	}

	/*SEARCH THIS END IN TABLE*/

	uint64_t st_a = 0;
	uint64_t en_a = chainAmount - 1;
	bool flag = false;
	uint8_t this_pass_l;

	do
	{
		uint64_t counter_chains = (st_a + en_a) / 2; //будет отвечать за середину между началом и концом
		bool compared = false;
		for (uint8_t size_1 = 0; (size_1 < maxPasswordLength) && (!compared); size_1++)
		{
			if (threadBlock_byteptr_shared[(size_1 / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + size_1 % SHARED_MEMORY_BANK_WIDTH] < RTEndsBlock_global[counter_chains * maxPasswordLength + size_1])
			{
				compared = true;
				if (counter_chains == 0)
					st_a = en_a + 1;
				else
					en_a = counter_chains - 1;
			}
			if (threadBlock_byteptr_shared[(size_1 / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + size_1 % SHARED_MEMORY_BANK_WIDTH] > RTEndsBlock_global[counter_chains * maxPasswordLength + size_1])
			{
				compared = true;
				if (counter_chains == chainAmount - 1)
					st_a = en_a + 1;
				else
					st_a = counter_chains + 1;
			}
			if (threadBlock_byteptr_shared[(size_1 / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + size_1 % SHARED_MEMORY_BANK_WIDTH] == RTEndsBlock_global[counter_chains * maxPasswordLength + size_1])
			{
				if (RTEndsBlock_global[counter_chains * maxPasswordLength + size_1] == '\0')
				{
					this_pass_l = size_1;
					break;
				}
				continue;
			}
		}
		if (!compared)
		{
			if (this_pass_l == 0)
				this_pass_l = maxPasswordLength;

			/*HASH FOUND PROBABLY IN THIS TYPE OF LINE; NEED TO SEARCH STRING*/

			while (counter_chains < chainAmount)
			{
				uint8_t size_1;
				for (size_1 = 0; size_1 < this_pass_l; size_1++)
				{
					if (threadBlock_byteptr_shared[(size_1 / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + size_1 % SHARED_MEMORY_BANK_WIDTH] == RTEndsBlock_global[counter_chains * maxPasswordLength + size_1])
						continue;
					else
						break;
				}
				if (size_1 != this_pass_l)
					break;
				counter_chains--;
			}
			
			if (counter_chains > chainAmount)
				counter_chains = 0;
			else
				counter_chains++;

			/*SEARCH IN LINES WITH SAME END*/

			while ((!flag) && (counter_chains < chainAmount))
			{
				/* check if this end is legit */

				uint8_t size_1;
				for (size_1 = 0; size_1 < this_pass_l; size_1++)
				{
					if (threadBlock_byteptr_shared[(size_1 / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + size_1 % SHARED_MEMORY_BANK_WIDTH] == RTEndsBlock_global[counter_chains * maxPasswordLength + size_1])
						continue;
					else
						break;
				}
				if (size_1 != this_pass_l)
					break;

				sha256_init_pass(threadBlock_byteptr_shared, &(RTStartsBlock_global[counter_chains * maxPasswordLength]), maxPasswordLength);

				for (uint32_t i = 0; i < chainLength - localChainLength - 1; i++)
				{ //  A password -> hash -> password iteration.
					/**passwordLength = 3;
					threadBlock_shared[0] = 'a';
					threadBlock_shared[1] = 'b';
					threadBlock_shared[2] = 'c';*/

					// Transform password into a valid for sha256 512-bit block.
					make_full_block(threadBlock_byteptr_shared, maxPasswordLength);

					// Hash the password.
					sha256_init_hash(threadHash_shared);
					sha256_transform(threadHash_shared, (const uint32_t *)threadBlock_shared);
					// Now state contains a sha256 hash of the password.

					// Perform reduction.

					if (mode)
						reduction_human(threadHash_shared, minPasswordLength, maxPasswordLength, i, mtableStatsNums_global, mtableStatsRating_global, mtableChar_global, threadBlock_byteptr_shared, padding, humanity);
					else
						reduction_rand(threadHash_shared, minPasswordLength, maxPasswordLength, i, threadBlock_byteptr_shared, padding);

					// Now threadBlock_shared contains a password made with reduction
				}
				for (uint8_t i = 0; i < maxPasswordLength; i++)
					threadBlock_global[i] = threadBlock_byteptr_shared[(i / SHARED_MEMORY_BANK_WIDTH) * SHARED_MEMORY_LINE_WIDTH + i % SHARED_MEMORY_BANK_WIDTH];

				// Transform password into a valid for sha256 512-bit block.
				make_full_block(threadBlock_byteptr_shared, maxPasswordLength);

				// Hash the password.
				sha256_init_hash(threadHash_shared);
				sha256_transform(threadHash_shared, (const uint32_t *)threadBlock_shared);

				flag = true;
				for (uint8_t i = 0; (i < 8) && (flag); i++)
					if (threadHash_global[i] != threadHash_shared[i * SHARED_MEMORY_BANK_COUNT])
						flag = false;

				if (flag)
					continue;
				else
				{
					for (uint8_t i = 0; i < maxPasswordLength; i++)
						threadBlock_global[i] = '\0';
					counter_chains++;
				}	
			}
			break;
		}
	} while ((st_a <= en_a) && (!flag));
}

int Generate_RT_CUDA(rainbow_t *rt, int blocksNum, int threadsNum, bool mode) //true - human, false - random
{
	/*
	const uint32_t host_sha256_h[8] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
	0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
	};

	const uint32_t host_sha256_k[64] = {
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
		0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
		0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
		0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
		0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
		0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
		0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
		0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
		0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
		0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
		0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
		0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
	};
	*/

	//system("cls");

		/*PRE-CALCULATIONS*/

	/* set up grid configuration */

	dim3 gridSize(blocksNum, 1, 1);
	dim3 blockSize(threadsNum, 1, 1);
	uint16_t normalizedBlockSize;

	uint64_t chain_am = rt->getChain_am_part() - rt->getTemp_size();
	uint64_t temp_chain_am = chain_am;
	uint16_t bigLoopCount = (uint16_t)(temp_chain_am / (gridSize.x * blockSize.x)); //full block quantity
	temp_chain_am -= bigLoopCount * (gridSize.x * blockSize.x);
	uint16_t newBlockCount = (uint16_t)((temp_chain_am / SM_COUNT / threadsNum) * SM_COUNT); //
	temp_chain_am -= newBlockCount * threadsNum;
	uint16_t newThreadsCount = (uint16_t)(temp_chain_am / SM_COUNT);
	temp_chain_am -= SM_COUNT * newThreadsCount;
	uint16_t remainingThreads = (uint16_t)temp_chain_am;
	/*
	cout << "Calculations: ";
	cout << bigLoopCount << "x" << blocksNum << "x" << threadsNum;
	cout << "+";
	cout << newBlockCount << "x" << threadsNum;
	cout << "+";
	cout << SM_COUNT << "x" << newThreadsCount;
	cout << "+";
	cout << "1x" << remainingThreads << "\t\t" << endl;
	*/
	if ((bigLoopCount * blocksNum * threadsNum + newBlockCount * threadsNum + SM_COUNT * newThreadsCount + remainingThreads) != chain_am)
	{
		cout << "WTF??" << endl;
		return 0;
	}

	/* calculate amount of required shared memory per block */

	size_t sharedMemoryPerBlock = 0;
	sharedMemoryPerBlock += blockSize.x * 8 * sizeof(uint32_t); //Hashes.
	sharedMemoryPerBlock += blockSize.x * 16 * sizeof(uint32_t); // Plain / 512-bit block
	if (sharedMemoryPerBlock % 128 > 0)
	{
		sharedMemoryPerBlock = (sharedMemoryPerBlock / SHARED_MEMORY_LINE_WIDTH + 1) * SHARED_MEMORY_LINE_WIDTH;
	}

		/*ALLOCATE RAM*/

	uint8_t max_l = rt->getMax_l();
	char *plainBlock_host = (char *)calloc(max_l * chain_am, sizeof(char));

		/*ALLOCATE ALL MT RELATED MEMORY*/

	markov3_t *mt;
	uint16_t *mtableStatsNums_host;
	uint8_t *mtableStatsRating_host;
	char *mtableChars_host;
	uint16_t *mtableStatsNums_global;
	uint8_t *mtableStatsRating_global;
	char *mtableChars_global;
	uint8_t humanity = rt->getHumanity();

	if (mode)
	{
		mtableStatsNums_host = (uint16_t *)calloc(MAX_PASS_L * 95 * 95, sizeof(uint16_t));
		mtableStatsRating_host = (uint8_t *)calloc(MAX_PASS_L * 95 * 95 * 95, sizeof(uint8_t));
		mtableChars_host = (char *)calloc(MAX_PASS_L * 95 * 95 * 95, sizeof(char));

		/*INITIALIZE MTABLE*/

		mt = rt->getMTptr();

		for (uint8_t i = 0; i < MAX_PASS_L; i++)
			for (uint8_t j = 0; j < 95; j++)
				for (uint8_t k = 0; k < 95; k++)
				{
					mtableStatsNums_host[i * 95 * 95 + j * 95 + k] = mt->stats_nums->at(i)->at(j)->at(k);
					for (uint8_t n = 0; n < 95; n++)
					{
						mtableStatsRating_host[i * 95 * 95 * 95 + j * 95 * 95 + k * 95 + n] = mt->stats_rating->at(i)->at(j)->at(k)->at(n);
						mtableChars_host[i * 95 * 95 * 95 + j * 95 * 95 + k * 95 + n] = mt->stats_char->at(i)->at(j)->at(k)->at(n);
					}
				}

		//cout << "Markov table (RAM) initialized...\t\t\t\t\r";

		checkCudaErrors(cudaMalloc((void **)&mtableStatsNums_global, MAX_PASS_L * 95 * 95 * sizeof(uint16_t)));
		checkCudaErrors(cudaMalloc((void **)&mtableStatsRating_global, MAX_PASS_L * 95 * 95 * 95 * sizeof(uint8_t)));
		checkCudaErrors(cudaMalloc((void **)&mtableChars_global, MAX_PASS_L * 95 * 95 * 95 * sizeof(char)));
		
		/* copy mtable */

		checkCudaErrors(cudaMemcpy(mtableStatsNums_global, mtableStatsNums_host, MAX_PASS_L * 95 * 95 * sizeof(uint16_t), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(mtableStatsRating_global, mtableStatsRating_host, MAX_PASS_L * 95 * 95 * 95 * sizeof(uint8_t), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(mtableChars_global, mtableChars_host, MAX_PASS_L * 95 * 95 * 95 * sizeof(char), cudaMemcpyHostToDevice));

		//cout << "Constants and markov table (VRAM) copied...\t\t\r";

		/* immediately free mtable_host */

		// "We don't need it here anymore"

		free(mtableStatsNums_host);
		free(mtableStatsRating_host);
		free(mtableChars_host);

		//cout << "Markov table (RAM) cleared...\t\t\t\t\r";
	}

	//cout << "RAM allocated...\t\t\t\t\t\r";

		/*ALLOCATE VRAM*/

	char *plainBlock_global;
	checkCudaErrors(cudaMalloc((void **)&plainBlock_global, chain_am * max_l * sizeof(char)));

	//cout << "VRAM allocated...\t\t\t\t\t\r";

		/*COPY ALL POSSIBLE FROM RAM TO VRAM*/

	/* copy constants */
	/*
	checkCudaErrors(cudaMemcpyToSymbol(sha256_h, host_sha256_h, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(sha256_k, host_sha256_k, 64 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
	*/	

		/*FINAL PREPARATIONS*/

	/* rewrite start/end array and allocate it in RAM */

		// This array is supposed to be start and end at the same time.
		// Starts already are stored in this->starts.

	uint8_t str_size;
	string str;
	uint64_t temp_size = rt->getTemp_size();

	for (uint64_t cntr = 0; cntr < chain_am; cntr++)
	{
		str = rt->getStart(cntr + temp_size);
		str_size = (uint8_t)(str.size());
		for (uint8_t i = 0; i < str_size; i++)
			plainBlock_host[cntr * max_l + i] = str.at(i);
		for (uint8_t i = str_size; i < max_l; i++)
			plainBlock_host[cntr * max_l + i] = '\0';
	}

	checkCudaErrors(cudaMemcpy(plainBlock_global, plainBlock_host, chain_am * max_l * sizeof(char), cudaMemcpyHostToDevice));

	//cout << "Plain array copied...\t\t\t\t\t\r";

	/* create events to calculate gen_dur */

	cudaEvent_t genStart, genStop;
	cudaEventCreate(&genStart);
	cudaEventCreate(&genStop);
	float totalElapsedTime = 0;

	/*GENERATE RT*/

	cudaError_t err;
	uint8_t min_l = rt->getMin_l();
	uint32_t chain_l = rt->getChain_l();
	uint8_t part_num = rt->getPart();
	uint32_t padding = chain_l * part_num;
	//uint32_t padding = 0;
	char *plainBlock_temp_global = &(plainBlock_global[0]);
	cudaEventRecord(genStart);

	/* first big part */

	for (int k = 0; k < bigLoopCount; k++)
	{
		plainBlock_temp_global = &(plainBlock_global[k * blockSize.x * gridSize.x * max_l]);

		/* run kernel */

		rainbowKernel_generate<<<gridSize, blockSize, sharedMemoryPerBlock>>>(min_l, max_l, mtableStatsNums_global, mtableStatsRating_global, mtableChars_global, chain_l, plainBlock_temp_global, mode, padding, humanity);

		//cout << "Kernel " << k + 1 << "/" << bigLoopCount << " done!\t\t\t\r\r";
		
		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error: %s\n", cudaGetErrorString(err));
		
		checkCudaErrors(cudaDeviceSynchronize());

		//cout << "Done! " << k + 1 << "/" << bigLoopCount << "\t\t\t\t\t\r";
	}
	//cout << endl;

	/* second part */

	if (newBlockCount)
	{
		plainBlock_temp_global = &(plainBlock_global[bigLoopCount * blockSize.x * gridSize.x * max_l]);
		gridSize.x = newBlockCount;

		rainbowKernel_generate<<<gridSize, blockSize, sharedMemoryPerBlock>>>(min_l, max_l, mtableStatsNums_global, mtableStatsRating_global, mtableChars_global, chain_l, plainBlock_temp_global, mode, padding, humanity);

		//cout << "Sub Kernel 1 done!\t\t\t\t\t\r";

		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error: %s\n", cudaGetErrorString(err));

		checkCudaErrors(cudaDeviceSynchronize());
	}

	/* third part */

	if (newThreadsCount)
	{
		plainBlock_temp_global = &(plainBlock_global[bigLoopCount * blockSize.x * blocksNum * max_l + blockSize.x * newBlockCount * max_l]);

		blockSize.x = newThreadsCount;
		if (blockSize.x % 32)
			normalizedBlockSize = (blockSize.x / 32 + 1) * 32;
		else
			normalizedBlockSize = blockSize.x;
		gridSize.x = SM_COUNT;
		sharedMemoryPerBlock = 0;
		sharedMemoryPerBlock += normalizedBlockSize * 8 * sizeof(uint32_t); //Hashes.
		sharedMemoryPerBlock += normalizedBlockSize * 16 * sizeof(uint32_t); // Plain / 512-bit block
		if ((sharedMemoryPerBlock % 128) > 0)
		{
			sharedMemoryPerBlock = (sharedMemoryPerBlock / SHARED_MEMORY_LINE_WIDTH + 1) * SHARED_MEMORY_LINE_WIDTH;
		}

		rainbowKernel_generate<<<gridSize, blockSize, sharedMemoryPerBlock>>>(min_l, max_l, mtableStatsNums_global, mtableStatsRating_global, mtableChars_global, chain_l, plainBlock_temp_global, mode, padding, humanity);

		//cout << "Sub Kernel 2 done!\t\t\t\t\t\r";

		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error: %s\n", cudaGetErrorString(err));

		checkCudaErrors(cudaDeviceSynchronize());
	}

	/* fourth part */

	if (remainingThreads)
	{
		plainBlock_temp_global = &(plainBlock_global[bigLoopCount * threadsNum * blocksNum * max_l + threadsNum * newBlockCount * max_l + newThreadsCount * SM_COUNT * max_l]);

		blockSize.x = remainingThreads;
		if (blockSize.x % 32)
			normalizedBlockSize = (blockSize.x / 32 + 1) * 32;
		else
			normalizedBlockSize = blockSize.x;
		gridSize.x = 1;
		sharedMemoryPerBlock = 0;
		sharedMemoryPerBlock += normalizedBlockSize * 8 * sizeof(uint32_t); //Hashes.
		sharedMemoryPerBlock += normalizedBlockSize * 16 * sizeof(uint32_t); // Plain / 512-bit block
		if ((sharedMemoryPerBlock % 128) > 0)
		{
			sharedMemoryPerBlock = (sharedMemoryPerBlock / SHARED_MEMORY_LINE_WIDTH + 1) * SHARED_MEMORY_LINE_WIDTH;
		}

		rainbowKernel_generate<<<gridSize, blockSize, sharedMemoryPerBlock>>>(min_l, max_l, mtableStatsNums_global, mtableStatsRating_global, mtableChars_global, chain_l, plainBlock_temp_global, mode, padding, humanity);

		//cout << "Sub Kernel 3 done!\t\t\t\t\t\r";

		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error: %s\n", cudaGetErrorString(err));

		checkCudaErrors(cudaDeviceSynchronize());
	}

		/*FINISH ALL*/

	/* stop timing */

	cudaEventRecord(genStop);
	cudaEventSynchronize(genStop);
	cudaEventElapsedTime(&totalElapsedTime, genStart, genStop);
	uint64_t totalElapsedTimeInMilliseconds = (uint64_t)totalElapsedTime;
	rt->setGenDur(totalElapsedTimeInMilliseconds);
	uint64_t secondsElapsed = totalElapsedTimeInMilliseconds / 1000;
	uint64_t minutesElapsed = secondsElapsed / 60;
	uint64_t hoursElapsed = minutesElapsed / 60;
	minutesElapsed -= hoursElapsed * 60;
	secondsElapsed -= hoursElapsed * 3600 + minutesElapsed * 60;
	totalElapsedTimeInMilliseconds %= 1000;

	string elapsedTime = "";
	elapsedTime += std::to_string(hoursElapsed) + " h " + std::to_string(minutesElapsed) + " m " + std::to_string(secondsElapsed) + " s " + std::to_string(totalElapsedTimeInMilliseconds) + " ms\t\t\t";

	//cout << "GEN TIME:\t" << elapsedTime << endl;

	/* get chain ends */

	checkCudaErrors(cudaMemcpy(plainBlock_host, plainBlock_global, chain_am * max_l * sizeof(char), cudaMemcpyDeviceToHost));

	/* copy them to rt_ptr */

	//cout << "Copying end vector...\t\t\t\t";
	for (uint64_t k = 0; k < chain_am; k++)
	{
		str.assign(max_l, '\0');
		int i;
		for (i = 0; (i < max_l) && (plainBlock_host[k * max_l + i] != '\0'); i++)
			str.at(i) = plainBlock_host[k * max_l + i];
		str.resize(i);
		rt->pushBackEnd(str);
		/*
		if (!(k % 10000))
			cout << "\rCopying end vector..." << k << "/" << chain_am << " done\t\t\t";
		*/
	}	

	//cout << "\rALL DONE!\t\t\t\t\t" << endl;

	checkCudaErrors(cudaFree(plainBlock_global));

	if (mode)
	{
		checkCudaErrors(cudaFree(mtableStatsNums_global));
		checkCudaErrors(cudaFree(mtableStatsRating_global));
		checkCudaErrors(cudaFree(mtableChars_global));
	}

	free(plainBlock_host);

	return 0;
}

vector<string>* Search_Prob_CUDA(rainbow_t *rt, uint32_t *hash_array, uint16_t amount, int blocksNum, int threadsNum)
{
	bool mode;
	if (rt->getMTptr())
		mode = true;
	else
		mode = false;

		/*PRE-CALCULATIONS*/

	/* set up grid configuration */

	dim3 gridSize(blocksNum, 1, 1);
	dim3 blockSize(threadsNum, 1, 1);
	uint16_t normalizedBlockSize;

	uint64_t chain_am = amount * rt->getChain_l();

	uint64_t temp_chain_am = chain_am;
	uint16_t bigLoopCount = (uint16_t)(temp_chain_am / (gridSize.x * blockSize.x));
	temp_chain_am -= bigLoopCount * (gridSize.x * blockSize.x);
	uint16_t newBlockCount = (uint16_t)((temp_chain_am / SM_COUNT / threadsNum) * SM_COUNT);
	temp_chain_am -= newBlockCount * threadsNum;
	uint16_t newThreadsCount = (uint16_t)(temp_chain_am / SM_COUNT);
	temp_chain_am -= SM_COUNT * newThreadsCount;
	uint16_t remainingThreads = (uint16_t)temp_chain_am;

	if ((bigLoopCount * blocksNum * threadsNum + newBlockCount * threadsNum + SM_COUNT * newThreadsCount + remainingThreads) != chain_am)
	{
		cout << "WTF??" << endl;
		return 0;
	}

	/* calculate amount of required shared memory per block */

	size_t sharedMemoryPerBlock = 0;
	sharedMemoryPerBlock += blockSize.x * 8 * sizeof(uint32_t); //Hashes.
	sharedMemoryPerBlock += blockSize.x * 16 * sizeof(uint32_t); // Plain / 512-bit block
	if (sharedMemoryPerBlock % 128 > 0)
	{
		sharedMemoryPerBlock = (sharedMemoryPerBlock / SHARED_MEMORY_LINE_WIDTH + 1) * SHARED_MEMORY_LINE_WIDTH;
	}

		/*ALLOCATE RAM*/

	uint8_t max_l = rt->getMax_l();
	char *plainBlock_host = (char *)calloc(max_l * chain_am, sizeof(char));

	markov3_t *mt;
	uint16_t *mtableStatsNums_host;
	uint8_t *mtableStatsRating_host;
	char *mtableChars_host;
	uint16_t *mtableStatsNums_global;
	uint8_t *mtableStatsRating_global;
	char *mtableChars_global;
	uint8_t humanity = rt->getHumanity();

	if (mode)
	{
		mtableStatsNums_host = (uint16_t *)calloc(MAX_PASS_L * 95 * 95, sizeof(uint16_t));
		mtableStatsRating_host = (uint8_t *)calloc(MAX_PASS_L * 95 * 95 * 95, sizeof(uint8_t));
		mtableChars_host = (char *)calloc(MAX_PASS_L * 95 * 95 * 95, sizeof(char));

		/*INITIALIZE MTABLE*/

		mt = rt->getMTptr();

		for (uint8_t i = 0; i < MAX_PASS_L; i++)
			for (uint8_t j = 0; j < 95; j++)
				for (uint8_t k = 0; k < 95; k++)
				{
					mtableStatsNums_host[i * 95 * 95 + j * 95 + k] = mt->stats_nums->at(i)->at(j)->at(k);
					for (uint8_t n = 0; n < 95; n++)
					{
						mtableStatsRating_host[i * 95 * 95 * 95 + j * 95 * 95 + k * 95 + n] = mt->stats_rating->at(i)->at(j)->at(k)->at(n);
						mtableChars_host[i * 95 * 95 * 95 + j * 95 * 95 + k * 95 + n] = mt->stats_char->at(i)->at(j)->at(k)->at(n);
					}
				}

		//cout << "Markov table (RAM) initialized...\t\t\t\t\r";

		checkCudaErrors(cudaMalloc((void **)&mtableStatsNums_global, MAX_PASS_L * 95 * 95 * sizeof(uint16_t)));
		checkCudaErrors(cudaMalloc((void **)&mtableStatsRating_global, MAX_PASS_L * 95 * 95 * 95 * sizeof(uint8_t)));
		checkCudaErrors(cudaMalloc((void **)&mtableChars_global, MAX_PASS_L * 95 * 95 * 95 * sizeof(char)));

		/* copy mtable */

		checkCudaErrors(cudaMemcpy(mtableStatsNums_global, mtableStatsNums_host, MAX_PASS_L * 95 * 95 * sizeof(uint16_t), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(mtableStatsRating_global, mtableStatsRating_host, MAX_PASS_L * 95 * 95 * 95 * sizeof(uint8_t), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(mtableChars_global, mtableChars_host, MAX_PASS_L * 95 * 95 * 95 * sizeof(char), cudaMemcpyHostToDevice));

		//cout << "Constants and markov table (VRAM) copied...\t\t\r";

		/* immediately free mtable_host */

		// "We don't need it here anymore"

		free(mtableStatsNums_host);
		free(mtableStatsRating_host);
		free(mtableChars_host);

		//cout << "Markov table (RAM) cleared...\t\t\t\t\r";
	}

	//cout << "RAM allocated...\t\t\t\t\t\r";

		/*ALLOCATE VRAM*/

	char *plainBlock_global;
	checkCudaErrors(cudaMalloc((void **)&plainBlock_global, chain_am * max_l * sizeof(char)));

	uint32_t *hashBlock_global;
	checkCudaErrors(cudaMalloc((void **)&hashBlock_global, amount * 8 * sizeof(uint32_t)));

	//cout << "VRAM allocated...\t\t\t\t\t\r";

		/*COPY ALL POSSIBLE FROM RAM TO VRAM*/

	/* copy constants */
	/*
	checkCudaErrors(cudaMemcpyToSymbol(sha256_h, host_sha256_h, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(sha256_k, host_sha256_k, 64 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
	*/

		/*FINAL PREPARATIONS*/

	/* initialize plainBlock with zeros (for to be sure) */

	for (uint64_t cntr = 0; cntr < chain_am; cntr++)
	{
		for (uint8_t i = 0; i < max_l; i++)
			plainBlock_host[cntr * max_l + i] = '\0';
	}

	checkCudaErrors(cudaMemcpy(plainBlock_global, plainBlock_host, chain_am * max_l * sizeof(char), cudaMemcpyHostToDevice));

	//cout << "Plain array copied...\t\t\t\t\t\r";

	/* copy hashes to global */

	checkCudaErrors(cudaMemcpy(hashBlock_global, hash_array, amount * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

	//cout << "Hash array copied...\t\t\t\t\t\r";

	/* create events to calculate gen_dur */
	/*
	cudaEvent_t genStart, genStop;
	cudaEventCreate(&genStart);
	cudaEventCreate(&genStop);
	float totalElapsedTime = 0;
	*/
		/*GENERATE CHAINS*/

	cudaError_t err;
	uint8_t min_l = rt->getMin_l();
	uint8_t part_num = rt->getPart();
	uint32_t chain_l = rt->getChain_l();
	uint32_t padding = chain_l * part_num;
	//uint32_t padding = 0;

	char *plainBlock_temp_global = &(plainBlock_global[0]);
	uint32_t calculated = 0;
	//cudaEventRecord(genStart);

	/* first big part */

	for (int k = 0; k < bigLoopCount; k++)
	{
		plainBlock_temp_global = &(plainBlock_global[k * blockSize.x * gridSize.x * max_l]);

		/* run kernel */

		rainbowKernel_probability<<<gridSize, blockSize, sharedMemoryPerBlock>>>(min_l, max_l, mtableStatsNums_global, mtableStatsRating_global, mtableChars_global, chain_l, plainBlock_temp_global, hashBlock_global, calculated, mode, padding, humanity);
		calculated += blockSize.x * gridSize.x;

		cout << "Kernel " << k + 1 << "/" << bigLoopCount << " done!\t\t\t\r\r";

		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error: %s\n", cudaGetErrorString(err));

		checkCudaErrors(cudaDeviceSynchronize());

		cout << "Done! " << k + 1 << "/" << bigLoopCount << "\t\t\t\t\t\r";
	}
	//cout << endl;

	/* second part */

	if (newBlockCount)
	{
		if (bigLoopCount)
			plainBlock_temp_global = &(plainBlock_global[bigLoopCount * blockSize.x * gridSize.x * max_l]);
		gridSize.x = newBlockCount;

		rainbowKernel_probability<<<gridSize, blockSize, sharedMemoryPerBlock>>>(min_l, max_l, mtableStatsNums_global, mtableStatsRating_global, mtableChars_global, chain_l, plainBlock_temp_global, hashBlock_global, calculated, mode, padding, humanity);
		calculated += blockSize.x * gridSize.x;

		cout << "Sub Kernel 1 done!\t\t\t\t\t\r";

		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error: %s\n", cudaGetErrorString(err));

		checkCudaErrors(cudaDeviceSynchronize());
	}

	/* third part */

	if (newThreadsCount)
	{
		if (bigLoopCount)
			plainBlock_temp_global = &(plainBlock_global[bigLoopCount * blockSize.x * blocksNum * max_l]);
		if (newBlockCount)
			plainBlock_temp_global = &(plainBlock_temp_global[blockSize.x * gridSize.x * max_l]);

		blockSize.x = newThreadsCount;
		if (blockSize.x % 32)
			normalizedBlockSize = (blockSize.x / 32 + 1) * 32;
		else
			normalizedBlockSize = blockSize.x;
		gridSize.x = SM_COUNT;
		sharedMemoryPerBlock = 0;
		sharedMemoryPerBlock += normalizedBlockSize * 8 * sizeof(uint32_t); //Hashes.
		sharedMemoryPerBlock += normalizedBlockSize * 16 * sizeof(uint32_t); // Plain / 512-bit block
		if ((sharedMemoryPerBlock % 128) > 0)
		{
			sharedMemoryPerBlock = (sharedMemoryPerBlock / SHARED_MEMORY_LINE_WIDTH + 1) * SHARED_MEMORY_LINE_WIDTH;
		}

		rainbowKernel_probability<<<gridSize, blockSize, sharedMemoryPerBlock>>>(min_l, max_l, mtableStatsNums_global, mtableStatsRating_global, mtableChars_global, chain_l, plainBlock_temp_global, hashBlock_global, calculated, mode, padding, humanity);
		calculated += blockSize.x * gridSize.x;

		cout << "Sub Kernel 2 done!\t\t\t\t\t\r";

		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error: %s\n", cudaGetErrorString(err));

		checkCudaErrors(cudaDeviceSynchronize());
	}

	/* fourth part */

	if (remainingThreads)
	{
		if (bigLoopCount)
			plainBlock_temp_global = &(plainBlock_global[bigLoopCount * threadsNum * blocksNum * max_l]);
		if (newBlockCount)
			plainBlock_temp_global = &(plainBlock_temp_global[threadsNum * newBlockCount * max_l]);
		if (newThreadsCount)
			plainBlock_temp_global = &(plainBlock_temp_global[newThreadsCount * SM_COUNT * max_l]);

		blockSize.x = remainingThreads;
		if (blockSize.x % 32)
			normalizedBlockSize = (blockSize.x / 32 + 1) * 32;
		else
			normalizedBlockSize = blockSize.x;
		gridSize.x = 1;
		sharedMemoryPerBlock = 0;
		sharedMemoryPerBlock += normalizedBlockSize * 8 * sizeof(uint32_t); //Hashes.
		sharedMemoryPerBlock += normalizedBlockSize * 16 * sizeof(uint32_t); // Plain / 512-bit block
		if ((sharedMemoryPerBlock % 128) > 0)
		{
			sharedMemoryPerBlock = (sharedMemoryPerBlock / SHARED_MEMORY_LINE_WIDTH + 1) * SHARED_MEMORY_LINE_WIDTH;
		}

		rainbowKernel_probability<<<gridSize, blockSize, sharedMemoryPerBlock>>>(min_l, max_l, mtableStatsNums_global, mtableStatsRating_global, mtableChars_global, chain_l, plainBlock_temp_global, hashBlock_global, calculated, mode, padding, humanity);
		calculated += blockSize.x * gridSize.x;

		cout << "Sub Kernel 3 done!\t\t\t\t\t\r";

		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error: %s\n", cudaGetErrorString(err));

		checkCudaErrors(cudaDeviceSynchronize());
	}

		/*FINISH ALL*/

	/* stop timing */
	/*
	cudaEventRecord(genStop);
	cudaEventSynchronize(genStop);
	cudaEventElapsedTime(&totalElapsedTime, genStart, genStop);
	uint64_t totalElapsedTimeInMilliseconds = (uint64_t)totalElapsedTime;
	rt_ptr->setGenDur(totalElapsedTimeInMilliseconds);
	uint64_t secondsElapsed = totalElapsedTimeInMilliseconds / 1000;
	uint64_t minutesElapsed = secondsElapsed / 60;
	uint64_t hoursElapsed = minutesElapsed / 60;
	minutesElapsed -= hoursElapsed * 60;
	secondsElapsed -= hoursElapsed * 3600 + minutesElapsed * 60;
	totalElapsedTimeInMilliseconds %= 1000;

	string elapsedTime = "";
	elapsedTime += std::to_string(hoursElapsed) + " h " + std::to_string(minutesElapsed) + " m " + std::to_string(secondsElapsed) + " s " + std::to_string(totalElapsedTimeInMilliseconds) + " ms\t\t\t";
	*/
	//cout << "GEN TIME:\t" << elapsedTime << endl;

	/* get chain ends */

	checkCudaErrors(cudaMemcpy(plainBlock_host, plainBlock_global, chain_am * max_l * sizeof(char), cudaMemcpyDeviceToHost));

	/* copy them to result vector */

	//cout << "Copying end vector...\t\t\t\t";

	vector<string> *result = new vector<string>();
	result->assign(amount * chain_l, "");
	string str;
	for (uint64_t k = 0; k < chain_am; k++)
	{
		str.assign(max_l, '\0');

		for (int i = 0; (i < max_l) && (plainBlock_host[k * max_l + i] < 127) && (plainBlock_host[k * max_l + i] > 31); i++)
			str.at(i) = plainBlock_host[k * max_l + i];
		result->at(k) = str;
		/*
		if (!(k % 10000))
			cout << "\rCopying end vector..." << k << "/" << chain_am << " done\t\t\t";
		*/
	}

	//cout << "\rALL DONE!\t\t\t\t\t" << endl;

	checkCudaErrors(cudaFree(plainBlock_global));
	checkCudaErrors(cudaFree(hashBlock_global));
	if (mode)
	{
		checkCudaErrors(cudaFree(mtableStatsNums_global));
		checkCudaErrors(cudaFree(mtableStatsRating_global));
		checkCudaErrors(cudaFree(mtableChars_global));
	}

	free(plainBlock_host);

	return result;
}

vector<string>* Search_Prob_Express_CUDA(rainbow_t *rt, uint32_t *hash_array, uint64_t amount, int blocksNum, int threadsNum)
{
	bool mode;
	if (rt->getMTptr())
		mode = true;
	else
		mode = false;

		/*PRE-CALCULATIONS*/

	/* set up grid configuration */

	dim3 gridSize(blocksNum, 1, 1);
	dim3 blockSize(threadsNum, 1, 1);
	uint16_t normalizedBlockSize;

	uint64_t chain_am = amount * rt->getChain_l();

	uint64_t temp_chain_am = chain_am;
	uint64_t bigLoopCount = (uint64_t)(temp_chain_am / (gridSize.x * blockSize.x));
	temp_chain_am -= bigLoopCount * (gridSize.x * blockSize.x);
	uint64_t newBlockCount = (uint64_t)((temp_chain_am / SM_COUNT / threadsNum) * SM_COUNT);
	temp_chain_am -= newBlockCount * threadsNum;
	uint64_t newThreadsCount = (uint64_t)(temp_chain_am / SM_COUNT);
	temp_chain_am -= SM_COUNT * newThreadsCount;
	uint64_t remainingThreads = (uint64_t)temp_chain_am;

	if ((bigLoopCount * blocksNum * threadsNum + newBlockCount * threadsNum + SM_COUNT * newThreadsCount + remainingThreads) != chain_am)
	{
		cout << "WTF??" << endl;
		return 0;
	}

	/* calculate amount of required shared memory per block */

	size_t sharedMemoryPerBlock = 0;
	sharedMemoryPerBlock += blockSize.x * 8 * sizeof(uint32_t); //Hashes.
	sharedMemoryPerBlock += blockSize.x * 16 * sizeof(uint32_t); // Plain / 512-bit block
	if (sharedMemoryPerBlock % 128 > 0)
	{
		sharedMemoryPerBlock = (sharedMemoryPerBlock / SHARED_MEMORY_LINE_WIDTH + 1) * SHARED_MEMORY_LINE_WIDTH;
	}

	/*ALLOCATE RAM*/

	uint8_t max_l = rt->getMax_l();
	uint64_t chain_am_part = rt->getChain_am_part();
	char *plainBlock_host = (char *)calloc(max_l * chain_am, sizeof(char));

	markov3_t *mt;
	uint16_t *mtableStatsNums_host;
	uint8_t *mtableStatsRating_host;
	char *mtableChars_host;
	uint16_t *mtableStatsNums_global;
	uint8_t *mtableStatsRating_global;
	char *mtableChars_global;
	uint8_t humanity = rt->getHumanity();

	if (mode)
	{
		mtableStatsNums_host = (uint16_t *)calloc(MAX_PASS_L * 95 * 95, sizeof(uint16_t));
		mtableStatsRating_host = (uint8_t *)calloc(MAX_PASS_L * 95 * 95 * 95, sizeof(uint8_t));
		mtableChars_host = (char *)calloc(MAX_PASS_L * 95 * 95 * 95, sizeof(char));

		/*INITIALIZE MTABLE*/

		mt = rt->getMTptr();

		for (uint8_t i = 0; i < MAX_PASS_L; i++)
			for (uint8_t j = 0; j < 95; j++)
				for (uint8_t k = 0; k < 95; k++)
				{
					mtableStatsNums_host[i * 95 * 95 + j * 95 + k] = mt->stats_nums->at(i)->at(j)->at(k);
					for (uint8_t n = 0; n < 95; n++)
					{
						mtableStatsRating_host[i * 95 * 95 * 95 + j * 95 * 95 + k * 95 + n] = mt->stats_rating->at(i)->at(j)->at(k)->at(n);
						mtableChars_host[i * 95 * 95 * 95 + j * 95 * 95 + k * 95 + n] = mt->stats_char->at(i)->at(j)->at(k)->at(n);
					}
				}

		//cout << "Markov table (RAM) initialized...\t\t\t\t\r";

		checkCudaErrors(cudaMalloc((void **)&mtableStatsNums_global, MAX_PASS_L * 95 * 95 * sizeof(uint16_t)));
		checkCudaErrors(cudaMalloc((void **)&mtableStatsRating_global, MAX_PASS_L * 95 * 95 * 95 * sizeof(uint8_t)));
		checkCudaErrors(cudaMalloc((void **)&mtableChars_global, MAX_PASS_L * 95 * 95 * 95 * sizeof(char)));

		/* copy mtable */

		checkCudaErrors(cudaMemcpy(mtableStatsNums_global, mtableStatsNums_host, MAX_PASS_L * 95 * 95 * sizeof(uint16_t), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(mtableStatsRating_global, mtableStatsRating_host, MAX_PASS_L * 95 * 95 * 95 * sizeof(uint8_t), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(mtableChars_global, mtableChars_host, MAX_PASS_L * 95 * 95 * 95 * sizeof(char), cudaMemcpyHostToDevice));

		//cout << "Constants and markov table (VRAM) copied...\t\t\r";

		/* immediately free mtable_host */

		// "We don't need it here anymore"

		free(mtableStatsNums_host);
		free(mtableStatsRating_host);
		free(mtableChars_host);

		//cout << "Markov table (RAM) cleared...\t\t\t\t\r";
	}

	char *RTEndsBlock_host = (char *)calloc(max_l * chain_am_part, sizeof(char));
	char *RTStartsBlock_host = (char *)calloc(max_l * chain_am_part, sizeof(char));
	string temp_str;

	for (uint64_t counter = 0; counter < chain_am_part; counter++)
	{
		temp_str = rt->getStart(counter);
		for (uint16_t subcntr = 0; subcntr < temp_str.size(); subcntr++)
			RTStartsBlock_host[counter * max_l + subcntr] = temp_str.at(subcntr);

		temp_str = rt->getEnd(counter);
		for (uint16_t subcntr = 0; subcntr < temp_str.size(); subcntr++)
			RTEndsBlock_host[counter * max_l + subcntr] = temp_str.at(subcntr);
	}

	//cout << "RAM allocated...\t\t\t\t\t\r";

		/*ALLOCATE VRAM*/

	char *plainBlock_global;
	checkCudaErrors(cudaMalloc((void **)&plainBlock_global, chain_am * max_l * sizeof(char)));

	char *RTStartsBlock_global;
	checkCudaErrors(cudaMalloc((void **)&RTStartsBlock_global, chain_am_part * max_l * sizeof(char)));

	char *RTEndsBlock_global;
	checkCudaErrors(cudaMalloc((void **)&RTEndsBlock_global, chain_am_part * max_l * sizeof(char)));

	uint32_t *hashBlock_global;
	checkCudaErrors(cudaMalloc((void **)&hashBlock_global, amount * 8 * sizeof(uint32_t)));

	//cout << "VRAM allocated...\t\t\t\t\t\r";

		/*COPY ALL POSSIBLE FROM RAM TO VRAM*/

	/* copy constants */
	/*
	checkCudaErrors(cudaMemcpyToSymbol(sha256_h, host_sha256_h, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(sha256_k, host_sha256_k, 64 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
	*/

		/*FINAL PREPARATIONS*/

	/* initialize plainBlock with zeros (for to be sure) */

	for (uint64_t cntr = 0; cntr < chain_am; cntr++)
	{
		for (uint8_t i = 0; i < max_l; i++)
			plainBlock_host[cntr * max_l + i] = '\0';
	}

	checkCudaErrors(cudaMemcpy(plainBlock_global, plainBlock_host, chain_am * max_l * sizeof(char), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(RTStartsBlock_global, RTStartsBlock_host, chain_am_part * max_l * sizeof(char), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(RTEndsBlock_global, RTEndsBlock_host, chain_am_part * max_l * sizeof(char), cudaMemcpyHostToDevice));

	/* immediately delete start and end blocks on host */

	free(RTStartsBlock_host);
	free(RTEndsBlock_host);

	//cout << "Plain array copied...\t\t\t\t\t\r";

	/* copy hashes to global */

	checkCudaErrors(cudaMemcpy(hashBlock_global, hash_array, amount * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

	//cout << "Hash array copied...\t\t\t\t\t\r";

	/* create events to calculate gen_dur */
	/*
	cudaEvent_t genStart, genStop;
	cudaEventCreate(&genStart);
	cudaEventCreate(&genStop);
	float totalElapsedTime = 0;
	*/
	/*GENERATE CHAINS*/

	cudaError_t err;
	uint8_t min_l = rt->getMin_l();
	uint8_t part_num = rt->getPart();
	uint32_t chain_l = rt->getChain_l();
	uint32_t padding = chain_l * part_num;
	//uint32_t padding = 0;

	char *plainBlock_temp_global = &(plainBlock_global[0]);
	uint64_t calculated = 0;
	//cudaEventRecord(genStart);

	/* first big part */

	for (int k = 0; k < bigLoopCount; k++)
	{
		plainBlock_temp_global = &(plainBlock_global[k * blockSize.x * gridSize.x * max_l]);

		/* run kernel */

		rainbowKernel_expr_probability<<<gridSize, blockSize, sharedMemoryPerBlock>>>(min_l, max_l, mtableStatsNums_global, mtableStatsRating_global, mtableChars_global, chain_l, chain_am_part, plainBlock_temp_global, RTStartsBlock_global, RTEndsBlock_global, hashBlock_global, calculated, mode, padding, humanity);
		calculated += blockSize.x * gridSize.x;

		cout << "Kernel " << k + 1 << "/" << bigLoopCount << " done!\t\t\t\r\r";

		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error: %s\n", cudaGetErrorString(err));

		checkCudaErrors(cudaDeviceSynchronize());

		cout << "Done! " << k + 1 << "/" << bigLoopCount << "\t\t\t\t\t\r";
	}
	//cout << endl;

	/* second part */

	if (newBlockCount)
	{
		if (bigLoopCount)
			plainBlock_temp_global = &(plainBlock_global[bigLoopCount * blockSize.x * gridSize.x * max_l]);
		gridSize.x = (uint32_t)newBlockCount;

		rainbowKernel_expr_probability<<<gridSize, blockSize, sharedMemoryPerBlock>>>(min_l, max_l, mtableStatsNums_global, mtableStatsRating_global, mtableChars_global, chain_l, chain_am_part, plainBlock_temp_global, RTStartsBlock_global, RTEndsBlock_global, hashBlock_global, calculated, mode, padding, humanity);
		calculated += blockSize.x * gridSize.x;

		cout << "Sub Kernel 1 done!\t\t\t\t\t\r";

		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error: %s\n", cudaGetErrorString(err));

		checkCudaErrors(cudaDeviceSynchronize());
	}

	/* third part */

	if (newThreadsCount)
	{
		if (bigLoopCount)
			plainBlock_temp_global = &(plainBlock_global[bigLoopCount * blockSize.x * blocksNum * max_l]);
		if (newBlockCount)
			plainBlock_temp_global = &(plainBlock_temp_global[blockSize.x * gridSize.x * max_l]);

		blockSize.x = (uint32_t)newThreadsCount;
		if (blockSize.x % 32)
			normalizedBlockSize = (blockSize.x / 32 + 1) * 32;
		else
			normalizedBlockSize = blockSize.x;
		gridSize.x = SM_COUNT;
		sharedMemoryPerBlock = 0;
		sharedMemoryPerBlock += normalizedBlockSize * 8 * sizeof(uint32_t); //Hashes.
		sharedMemoryPerBlock += normalizedBlockSize * 16 * sizeof(uint32_t); // Plain / 512-bit block
		if ((sharedMemoryPerBlock % 128) > 0)
		{
			sharedMemoryPerBlock = (sharedMemoryPerBlock / SHARED_MEMORY_LINE_WIDTH + 1) * SHARED_MEMORY_LINE_WIDTH;
		}

		rainbowKernel_expr_probability<<<gridSize, blockSize, sharedMemoryPerBlock>>>(min_l, max_l, mtableStatsNums_global, mtableStatsRating_global, mtableChars_global, chain_l, chain_am_part, plainBlock_temp_global, RTStartsBlock_global, RTEndsBlock_global, hashBlock_global, calculated, mode, padding, humanity);
		calculated += blockSize.x * gridSize.x;

		cout << "Sub Kernel 2 done!\t\t\t\t\t\r";

		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error: %s\n", cudaGetErrorString(err));

		checkCudaErrors(cudaDeviceSynchronize());
	}

	/* fourth part */

	if (remainingThreads)
	{
		if (bigLoopCount)
			plainBlock_temp_global = &(plainBlock_global[bigLoopCount * threadsNum * blocksNum * max_l]);
		if (newBlockCount)
			plainBlock_temp_global = &(plainBlock_temp_global[threadsNum * newBlockCount * max_l]);
		if (newThreadsCount)
			plainBlock_temp_global = &(plainBlock_temp_global[newThreadsCount * SM_COUNT * max_l]);

		blockSize.x = (uint32_t)remainingThreads;
		if (blockSize.x % 32)
			normalizedBlockSize = (blockSize.x / 32 + 1) * 32;
		else
			normalizedBlockSize = blockSize.x;
		gridSize.x = 1;
		sharedMemoryPerBlock = 0;
		sharedMemoryPerBlock += normalizedBlockSize * 8 * sizeof(uint32_t); //Hashes.
		sharedMemoryPerBlock += normalizedBlockSize * 16 * sizeof(uint32_t); // Plain / 512-bit block
		if ((sharedMemoryPerBlock % 128) > 0)
		{
			sharedMemoryPerBlock = (sharedMemoryPerBlock / SHARED_MEMORY_LINE_WIDTH + 1) * SHARED_MEMORY_LINE_WIDTH;
		}

		rainbowKernel_expr_probability<<<gridSize, blockSize, sharedMemoryPerBlock>>>(min_l, max_l, mtableStatsNums_global, mtableStatsRating_global, mtableChars_global, chain_l, chain_am_part, plainBlock_temp_global, RTStartsBlock_global, RTEndsBlock_global, hashBlock_global, calculated, mode, padding, humanity);
		calculated += blockSize.x * gridSize.x;

		cout << "Sub Kernel 3 done!\t\t\t\t\t\r";

		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error: %s\n", cudaGetErrorString(err));

		checkCudaErrors(cudaDeviceSynchronize());
	}

		/*FINISH ALL*/

	/* stop timing */
	/*
	cudaEventRecord(genStop);
	cudaEventSynchronize(genStop);
	cudaEventElapsedTime(&totalElapsedTime, genStart, genStop);
	uint64_t totalElapsedTimeInMilliseconds = (uint64_t)totalElapsedTime;
	rt_ptr->setGenDur(totalElapsedTimeInMilliseconds);
	uint64_t secondsElapsed = totalElapsedTimeInMilliseconds / 1000;
	uint64_t minutesElapsed = secondsElapsed / 60;
	uint64_t hoursElapsed = minutesElapsed / 60;
	minutesElapsed -= hoursElapsed * 60;
	secondsElapsed -= hoursElapsed * 3600 + minutesElapsed * 60;
	totalElapsedTimeInMilliseconds %= 1000;

	string elapsedTime = "";
	elapsedTime += std::to_string(hoursElapsed) + " h " + std::to_string(minutesElapsed) + " m " + std::to_string(secondsElapsed) + " s " + std::to_string(totalElapsedTimeInMilliseconds) + " ms\t\t\t";
	*/
	//cout << "GEN TIME:\t" << elapsedTime << endl;

	/* get chain ends */

	checkCudaErrors(cudaMemcpy(plainBlock_host, plainBlock_global, chain_am * max_l * sizeof(char), cudaMemcpyDeviceToHost));

	/* copy them to result vector */

	//cout << "Copying end vector...\t\t\t\t";

	vector<string> *result = new vector<string>();
	result->assign(amount, "");
	string str;
	for (uint64_t k = 0; k < amount; k++)
	{
		for (uint32_t j = 0; (j < chain_l) && (result->at(k) == ""); j++)
		{
			str.assign(max_l, '\0');
			int i = 0;
			for (i = 0; (i < max_l) && (plainBlock_host[k * chain_l * max_l + j * max_l + i] != '\0'); i++)
				str.at(i) = plainBlock_host[k * chain_l * max_l + j * max_l + i];
			str.resize(i);

			if (str.size() && (result->at(k) == "") && (str != ""))
				result->at(k) = str;
		}
		/*
		if (!(k % 10000))
			cout << "\rCopying end vector..." << k << "/" << chain_am << " done\t\t\t";
		*/
	}

	//cout << "\rALL DONE!\t\t\t\t\t" << endl;

	checkCudaErrors(cudaFree(plainBlock_global));
	checkCudaErrors(cudaFree(RTStartsBlock_global));
	checkCudaErrors(cudaFree(RTEndsBlock_global));
	checkCudaErrors(cudaFree(hashBlock_global));

	if (mode)
	{
		checkCudaErrors(cudaFree(mtableStatsNums_global));
		checkCudaErrors(cudaFree(mtableStatsRating_global));
		checkCudaErrors(cudaFree(mtableChars_global));
	}

	free(plainBlock_host);

	return result;
}