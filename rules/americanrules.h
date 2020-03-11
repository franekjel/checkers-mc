#pragma once
#include "rules.h"

class AmericanRules : public Rules
{
public:
    __device__ __host__ AmericanRules()
        : Rules(8, 48) {}
    __device__ __host__ void getMovesLightPos(char board[], Move moves[], int& captures, int8_t x, int8_t y, int& n) override;
    __device__ __host__ void getMovesDarkPos(char board[], Move moves[], int& captures, int8_t x, int8_t y, int& n) override;
    __device__ __host__ int checkResult(char board[]) override;
};
