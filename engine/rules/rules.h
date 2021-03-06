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

//One move - x1,y1,x2,y2
typedef int8_t Move[4];

/*TODO for better perf and debugging change to this
typedef struct Move
{
    int8_t x1, y1, x2, y2;
    int8_t operator[](int i)
    {
        switch (i)
        {
        case 0:
            return x1;
        case 1:
            return y1;
        case 2:
            return x2;
        case 3:
            return y2;
        }
    }
};
*/

/* class Rules isn't used. It only show functions and variables class should implement to be used as game rules.
 * since polymorphism reduced performance ~90% using templates on static classes is probably the best soltion (best I can think of)
 * example implementation in americanrules.h
*/
class Rules
{
public:
    /* Generates moves for light player. Parameters:
     * board - buffer boardSize*boardSize
     * moves - array for moves, ensure it is big enough
     * captures - number of capturing moves
     * x,y - used during captures, indicate position on pawn, which must move
     * n - position in moves[] (increments with each move)
     */
    template <int TBoardSize, int TMaxMoves>
    static __device__ __host__ void getMovesLightPos(char board[TBoardSize][TBoardSize], Move moves[TMaxMoves], int& captures, int8_t x, int8_t y, int& n);

    /* Generates moves for dark player. Parameters:
     * board - buffer boardSize*boardSize
     * moves - array for moves, ensure it is big enough
     * captures - number of capturing moves
     * x,y - used during captures, indicate position on pawn, which must move
     * n - position in moves[] (increments with each move)
     */
    template <int TBoardSize, int TMaxMoves>
    static __device__ __host__ void getMovesDarkPos(char board[TBoardSize][TBoardSize], Move moves[TMaxMoves], int& captures, int8_t x, int8_t y, int& n);

    /*
     * Do one move (one means no multiple capture).
     * If after move piece is on square for promotion function should promote
     * Return value is true if there was promotion or false if not
    */
    template <int TBoardSize>
    static __device__ __host__ bool doMove(char board[TBoardSize][TBoardSize], Move move);
};

/* Parameters for templates:
 * class Rules - game rules, class should implement methods as above
 * boardSize - size of boards ofc
 * maxMoves - //maximum number of moves (eg. in american checkers, roughly estimate is 48, 12 pawns * 4 directions)
 */

template <class TRules, int TBoardSize, int TMaxMoves>
__device__ __host__ int genMovesLight(char board[TBoardSize][TBoardSize], Move moves[TMaxMoves], int& captures)
{
    int n = 0;
    captures = 0;
    for (int8_t x = 0; x < TBoardSize; x++)
    {
        for (int8_t y = 0; y < TBoardSize; y++)
        {
            TRules::getMovesLightPos(board, moves, captures, x, y, n);
        }
    }
    return n;
}

template <class TRules, int TBoardSize, int TMaxMoves>
__device__ __host__ int genMovesDark(char board[TBoardSize][TBoardSize], Move moves[TMaxMoves], int& captures)
{
    int n = 0;
    captures = 0;
    for (int8_t x = 0; x < TBoardSize; x++)
    {
        for (int8_t y = 0; y < TBoardSize; y++)
        {
            TRules::getMovesDarkPos(board, moves, captures, x, y, n);
        }
    }
    return n;
}

template <int TMaxMoves>
__device__ __host__ int filterMoves(Move moves[TMaxMoves], int n)
{
    Move tab[TMaxMoves];
    int m = 0;
    for (int i = 0; i < n; i++)
    {
        if (moves[i][0] - moves[i][2] == 2 || moves[i][0] - moves[i][2] == -2)
        {
            memcpy(tab[m], moves[i], sizeof(Move));
            m++;
        }
    }
    for (int i = 0; i < m; i++)
        memcpy(moves[i], tab[i], sizeof(Move));
    return m;
}
