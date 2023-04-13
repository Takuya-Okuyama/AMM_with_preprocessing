#ifndef __COMMON_HEADER__
#define __COMMON_HEADER__

// Macro
#define DIM_M 32

#ifdef DEBUG
#define print_if_debugging(fmt, ...) printf(fmt, ##__VA_ARGS__);
#else
#define print_if_debugging(fmt, ...)
#endif

// Header
#include <cassert>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "host_memory.h"
#include "device_memory.h"

#endif
