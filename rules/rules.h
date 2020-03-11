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

//One move - from.x, from.y, to.x, to.y It have operator=
typedef int8_t Move[4];

enum RulesType
{
    AMERICAN_RULES
};

class Rules
{
public:
    const int boardSize;
    const int maxMoves; //maximum number of moves (eg. in american checkers 48, 12 pawns * 4 directions)

    __device__ __host__ Rules(int size, int moves)
        : boardSize(size), maxMoves(moves) {}

    virtual ~Rules() {}
    /* Generates moves for light player. Parameters:
     * board - buffer boardSize*boardSize
     * moves - array for moves, ensure it is big enought
     * captures - number of capturing moves
     * x,y - used during captures, indicate position on pawn, which must move
     * n - position in moves[] (increments with each move)
     */
    virtual __device__ __host__ void getMovesLightPos(char* board, Move moves[], int& captures, int8_t x, int8_t y, int& n) = 0;

    /* Generates moves for dark player. Parameters:
     * board - buffer boardSize*boardSize
     * moves - array for moves, ensure it is big enought
     * captures - number of capturing moves
     * x,y - used during captures, indicate position on pawn, which must move
     * n - position in moves[] (increments with each move)
     */
    virtual __device__ __host__ void getMovesDarkPos(char* board, Move moves[], int& captures, int8_t x, int8_t y, int& n) = 0;

    /* Check for game result on given board state
     * 0 - light win, 1 dark win, -1 - nothing
     */
    virtual __device__ __host__ int checkResult(char board[]) = 0;

    __device__ __host__ int genMovesLight(char board[], Move moves[], int& captures);
    __device__ __host__ int genMovesDark(char board[], Move moves[], int& captures);
    __device__ __host__ int filterMoves(Move moves[], int n);
};
