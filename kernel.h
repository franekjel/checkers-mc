#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <string.h>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

extern "C"
{
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
}

#include "rules/americanrules.h"
#include "rules/rules.h"

// for Testla T4 (2560 CUDA Cores)
#define THREADS 10
#define BLOCK 256

// find best move for light player on given board using GPU
void findMoveGPU(char* board, int timeout, int player, RulesType rules);
