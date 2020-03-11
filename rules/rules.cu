#include "rules.h"

__device__ __host__ int Rules::genMovesLight(char board[], Move moves[], int& captures)
{
    int n = 0;
    captures = 0;
    for (int8_t x = 0; x < boardSize; x++)
    {
        for (int8_t y = 0; y < boardSize; y++)
        {
            getMovesLightPos(board, moves, captures, x, y, n);
        }
    }
    return n;
}

__device__ __host__ int Rules::genMovesDark(char board[], Move moves[], int& captures)
{
    int n = 0;
    captures = 0;
    for (int8_t x = 0; x < boardSize; x++)
    {
        for (int8_t y = 0; y < boardSize; y++)
        {
            getMovesDarkPos(board, moves, captures, x, y, n);
        }
    }
    return n;
}

__device__ __host__ int Rules::filterMoves(Move moves[], int n)
{
    Move tab[48];
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
