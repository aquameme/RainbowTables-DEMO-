#pragma once

// CUDA includes.
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "rainbow_t.h"

// C includes.
#include <Windows.h>
#include <stdint.h>

// C++ includes.
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <utility>

using std::cout;
using std::endl;
using std::string;
using std::ofstream;

// Copy hash from global to shared.
__device__ void sha256_copy_hash(uint32_t *threadHash_shared, uint32_t *threadHash_global);

// Initialise thread hash state with starting constants.
__device__ void sha256_init_hash(uint32_t *threadHash_shared);

// Initialize thread password (threadBlock_shared) state from global memory
__device__ void sha256_init_pass(char *threadBlock_shared, char *threadBlock_global, const int maxPasswordLength);

/*
* Description:
* Transforms a password into a valid for SHA-256 512-bit block.
* WARNING:
* Data assumed to fit in 1 sha-256 block (<400+- bits ~ 50 bytes).
* In:
* passwordLength;
* threadBlock_shared - memory for a 512-bit block;
* Out:
* threadBlock_shared - block built with password.
*/
__device__ void make_full_block(char *threadBlock_shared, const int maxPasswordLength);

/*
* SHA256 block compression function.  The 256-bit hash state is transformed via
* the 512-bit input block to produce a new state. Modified for lower register use.
* In:
* threadHash_shared - initial state of the hash (256-bit);
* threadBlock_shared - block of the hashed message (512-bit).
* Out:
* threadHash_shared - final state of the hash after hashing a block.
*/
__device__ void sha256_transform(uint32_t *threadHash_shared, const uint32_t *threadBlock_shared);

/*
* Description:
* Reduction function. Transforms a hash into password.
* In:
* threadHash_shared - transformed hash (256-bit).
* mtable_stats_rating_global
* mtable_stats_nums_global
* mtable_char_global
* Out:
* threadPassword_shared - password itself. (address is same as threadBlock_shared's)
*/
__device__ void reduction_human(uint32_t *threadHash_shared, const int minPasswordLength, const int maxPasswordLength,
	uint32_t iterationNumber, const uint16_t *mtableStatsNums_global, const uint8_t *mtableStatsRating_global,
	const char *mtableChar_global, char *threadBlock_shared, uint32_t padding, uint8_t humanity);

__device__ void reduction_rand(uint32_t *threadHash_shared, const int minPasswordLength, const int maxPasswordLength,
	uint32_t iterationNumber, char *threadBlock_shared, uint32_t padding);

/*
* Description:
* Main table generating kernel.
* Computes chains and unloads results to global memory (VRAM).
*/
__global__ void rainbowKernel_generate(const int minPasswordLength, const int maxPasswordLength,
	const uint16_t *mtableStatsNums_global, const uint8_t *mtableStatsRating_global, const char *mtableChar_global,
	const int chainLength, char *plainBlock_global, bool mode, uint32_t padding, uint8_t humanity);

__global__ void rainbowKernel_probability(const int minPasswordLength, const int maxPasswordLength,
	const uint16_t *mtableStatsNums_global, const uint8_t *mtableStatsRating_global, const char *mtableChar_global,
	const int chainLength, char *plainBlock_global, uint32_t *hashBlock_global, uint64_t calculated, bool mode, uint32_t padding, uint8_t humanity);

__global__ void rainbowKernel_expr_probability(const int minPasswordLength, const int maxPasswordLength,
	const uint16_t *mtableStatsNums_global, const uint8_t *mtableStatsRating_global, const char *mtableChar_global,
	const int chainLength, uint64_t chainAmount, char *plainBlock_global, char *RTStartsBlock_global, char *RTEndsBlock_global,
	uint32_t *hashBlock_global, uint64_t calculated, bool mode, uint32_t padding, uint8_t humanity);

/*
* Description:
* Initialises memory, launches the kernel, gets results from VRAM.
*/
int Generate_RT_CUDA(rainbow_t *rt, int blocksNum, int threadsNum, bool mode); //true - human, false - random

vector<string> *Search_Prob_CUDA(rainbow_t *rt, uint32_t *hash_array, uint16_t amount, int blocksNum, int threadsNum);

vector<string> *Search_Prob_Express_CUDA(rainbow_t *rt, uint32_t *hash_array, uint64_t amount, int blocksNum, int threadsNum);